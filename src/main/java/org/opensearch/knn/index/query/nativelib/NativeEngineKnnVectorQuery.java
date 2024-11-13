/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.nativelib;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * {@link KNNQuery} executes approximate nearest neighbor search (ANN) on a segment level.
 * {@link NativeEngineKnnVectorQuery} executes approximate nearest neighbor search but gives
 * us the control to combine the top k results in each leaf and post process the results just
 * for k-NN query if required. This is done by overriding rewrite method to execute ANN on each leaf
 * {@link KNNQuery} does not give the ability to post process segment results.
 */
@Log4j2
@Getter
@RequiredArgsConstructor
public class NativeEngineKnnVectorQuery extends Query {

    private final KNNQuery knnQuery;
    private final QueryUtils queryUtils;
    private final boolean expandNestedDocs;

    @Override
    public Weight createWeight(IndexSearcher indexSearcher, ScoreMode scoreMode, float boost) throws IOException {
        final IndexReader reader = indexSearcher.getIndexReader();
        final KNNWeight knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, scoreMode, 1);
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        List<PerLeafResult> perLeafResults;
        RescoreContext rescoreContext = knnQuery.getRescoreContext();
        final int finalK = knnQuery.getK();
        if (rescoreContext == null) {
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, finalK);
        } else {
            boolean isShardLevelRescoringEnabled = KNNSettings.isShardLevelRescoringEnabledForDiskBasedVector(knnQuery.getIndexName());
            int dimension = knnQuery.getQueryVector().length;
            int firstPassK = rescoreContext.getFirstPassK(finalK, isShardLevelRescoringEnabled, dimension);
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, firstPassK);
            if (isShardLevelRescoringEnabled == true) {
                ResultUtil.reduceToTopK(perLeafResults, firstPassK);
            }

            StopWatch stopWatch = new StopWatch().start();
            perLeafResults = doRescore(indexSearcher, leafReaderContexts, knnWeight, perLeafResults, finalK);
            long rescoreTime = stopWatch.stop().totalTime().millis();
            log.debug("Rescoring results took {} ms. oversampled k:{}, segments:{}", rescoreTime, firstPassK, leafReaderContexts.size());
        }
        ResultUtil.reduceToTopK(perLeafResults, finalK);

        if (expandNestedDocs) {
            StopWatch stopWatch = new StopWatch().start();
            perLeafResults = retrieveAll(indexSearcher, leafReaderContexts, knnWeight, perLeafResults, rescoreContext == null);
            long time_in_millis = stopWatch.stop().totalTime().millis();
            if (log.isDebugEnabled()) {
                long totalNestedDocs = perLeafResults.stream().mapToLong(perLeafResult -> perLeafResult.getResult().size()).sum();
                log.debug("Expanding of nested docs took {} ms. totalNestedDocs:{} ", time_in_millis, totalNestedDocs);
            }
        }

        TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        for (int i = 0; i < perLeafResults.size(); i++) {
            topDocs[i] = ResultUtil.resultMapToTopDocs(perLeafResults.get(i).getResult(), leafReaderContexts.get(i).docBase);
        }

        TopDocs topK = TopDocs.merge(getTotalTopDoc(topDocs), topDocs);

        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(indexSearcher, scoreMode, boost);
        }
        return queryUtils.createDocAndScoreQuery(reader, topK).createWeight(indexSearcher, scoreMode, boost);
    }

    /**
     * When expandNestedDocs is set to true, additional nested documents are retrieved.
     * As a result, the total number of documents will exceed k.
     * Instead of relying on the k value, we must count the total number of documents
     * to accurately determine how many are in topDocs.
     * The theoretical maximum value this method could return is Integer.MAX_VALUE,
     * as a single shard cannot have more documents than Integer.MAX_VALUE.
     *
     * @param topDocs the top documents
     * @return the total number of documents in the topDocs
     */
    private int getTotalTopDoc(TopDocs[] topDocs) {
        if (expandNestedDocs == false) {
            return knnQuery.getK();
        }

        int sum = 0;
        for (TopDocs topDoc : topDocs) {
            sum += topDoc.scoreDocs.length;
        }
        return sum;
    }

    private List<PerLeafResult> retrieveAll(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        List<PerLeafResult> perLeafResults,
        boolean useQuantizedVectors
    ) throws IOException {
        List<Callable<PerLeafResult>> nestedQueryTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            int finalI = i;
            nestedQueryTasks.add(() -> {
                PerLeafResult perLeafResult = perLeafResults.get(finalI);
                if (perLeafResult.getResult().isEmpty()) {
                    return perLeafResult;
                }

                DocIdSetIterator allSiblings = queryUtils.getAllSiblings(
                    leafReaderContext,
                    perLeafResult.getResult().keySet(),
                    knnQuery.getParentsFilter(),
                    perLeafResult.getFilterBits()
                );

                final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                    .matchedDocsIterator(allSiblings)
                    .numberOfMatchedDocs(allSiblings.cost())
                    // setting to false because in re-scoring we want to do exact search on full precision vectors
                    .useQuantizedVectorsForSearch(useQuantizedVectors)
                    .k((int) allSiblings.cost())
                    .isParentHits(false)
                    .knnQuery(knnQuery)
                    .build();
                Map<Integer, Float> rescoreResult = knnWeight.exactSearch(leafReaderContext, exactSearcherContext);
                perLeafResult.setResult(rescoreResult);
                return perLeafResult;
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(nestedQueryTasks);
    }

    private List<PerLeafResult> doSearch(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        int k
    ) throws IOException {
        List<Callable<PerLeafResult>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            tasks.add(() -> searchLeaf(leafReaderContext, knnWeight, k));
        }
        return indexSearcher.getTaskExecutor().invokeAll(tasks);
    }

    private List<PerLeafResult> doRescore(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        List<PerLeafResult> perLeafResults,
        int k
    ) throws IOException {
        List<Callable<PerLeafResult>> rescoreTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            int finalI = i;
            rescoreTasks.add(() -> {
                PerLeafResult perLeafeResult = perLeafResults.get(finalI);
                if (perLeafeResult.getResult().isEmpty()) {
                    return perLeafeResult;
                }
                DocIdSetIterator matchedDocs = ResultUtil.resultMapToDocIds(perLeafeResult.getResult());
                final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                    .matchedDocsIterator(matchedDocs)
                    .numberOfMatchedDocs(perLeafResults.get(finalI).getResult().size())
                    // setting to false because in re-scoring we want to do exact search on full precision vectors
                    .useQuantizedVectorsForSearch(false)
                    .k(k)
                    .isParentHits(false)
                    .knnQuery(knnQuery)
                    .build();
                Map<Integer, Float> rescoreResult = knnWeight.exactSearch(leafReaderContext, exactSearcherContext);
                perLeafeResult.setResult(rescoreResult);
                return perLeafeResult;
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks);
    }

    private PerLeafResult searchLeaf(LeafReaderContext ctx, KNNWeight queryWeight, int k) throws IOException {
        final PerLeafResult perLeafResult = queryWeight.searchLeaf(ctx, k);
        final Bits liveDocs = ctx.reader().getLiveDocs();
        if (liveDocs != null) {
            perLeafResult.getResult().entrySet().removeIf(entry -> liveDocs.get(entry.getKey()) == false);
        }
        return perLeafResult;
    }

    @Override
    public String toString(String field) {
        return this.getClass().getSimpleName() + "[" + field + "]..." + KNNQuery.class.getSimpleName() + "[" + knnQuery.toString() + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public boolean equals(Object obj) {
        if (!sameClassAs(obj)) {
            return false;
        }
        return knnQuery == ((NativeEngineKnnVectorQuery) obj).knnQuery;
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), knnQuery.hashCode());
    }
}
