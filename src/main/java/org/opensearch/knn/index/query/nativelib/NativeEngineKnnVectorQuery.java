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
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.TopDocsDISI;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.profile.LongMetric;
import org.opensearch.knn.profile.query.KNNMetrics;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.profile.AbstractProfileBreakdown;
import org.opensearch.search.profile.Profilers;
import org.opensearch.search.profile.Timer;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

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
        QueryProfiler profiler = ((ContextIndexSearcher) indexSearcher).getProfiler();
        if (profiler != null) {
            // add a new node to the profile tree
            profiler.getQueryBreakdown(knnQuery);
        }
        final KNNWeight knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, scoreMode, 1);
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        List<PerLeafResult> perLeafResults;
        RescoreContext rescoreContext = knnQuery.getRescoreContext();
        final int finalK = knnQuery.getK();
        if (rescoreContext == null || !rescoreContext.isRescoreEnabled()) {
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, finalK);
        } else {
            boolean isShardLevelRescoringDisabled = KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(knnQuery.getIndexName());
            int dimension = knnQuery.getQueryVector().length;
            int firstPassK = rescoreContext.getFirstPassK(finalK, isShardLevelRescoringDisabled, dimension);
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, firstPassK);
            if (isShardLevelRescoringDisabled == false) {
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
                long totalNestedDocs = perLeafResults.stream().mapToLong(perLeafResult -> perLeafResult.getResult().scoreDocs.length).sum();
                log.debug("Expanding of nested docs took {} ms. totalNestedDocs:{} ", time_in_millis, totalNestedDocs);
            }
        }

        TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        for (int i = 0; i < perLeafResults.size(); i++) {
            TopDocs leafTopDocs = perLeafResults.get(i).getResult();
            for (ScoreDoc scoreDoc : leafTopDocs.scoreDocs) {
                scoreDoc.doc += leafReaderContexts.get(i).docBase;
            }
            topDocs[i] = leafTopDocs;
        }

        TopDocs topK = TopDocs.merge(getTotalTopDoc(topDocs), topDocs);

        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(indexSearcher, scoreMode, boost);
        }
        return queryUtils.createDocAndScoreQuery(reader, topK, knnWeight).createWeight(indexSearcher, scoreMode, boost);
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
            QueryProfiler profiler = ((ContextIndexSearcher) indexSearcher).getProfiler();
            int finalI = i;
            nestedQueryTasks.add(() -> {
                if (profiler != null) {
                    AbstractProfileBreakdown profile = profiler.getProfileBreakdown(this).context(leafReaderContext);
                    PerLeafResult result = retrieveSingle(leafReaderContext, knnWeight, perLeafResults, useQuantizedVectors, finalI);
                    LongMetric metric = (LongMetric) profile.getMetric(KNNMetrics.NUM_NESTED_DOCS);
                    metric.setValue((long) result.getResult().scoreDocs.length);
                    return result;
                }
                return retrieveSingle(leafReaderContext, knnWeight, perLeafResults, useQuantizedVectors, finalI);
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(nestedQueryTasks);
    }

    private PerLeafResult retrieveSingle(
        LeafReaderContext leafReaderContext,
        KNNWeight knnWeight,
        List<PerLeafResult> perLeafResults,
        boolean useQuantizedVectors,
        int finalI
    ) throws IOException {
        PerLeafResult perLeafResult = perLeafResults.get(finalI);
        if (perLeafResult.getResult().scoreDocs.length == 0) {
            return perLeafResult;
        }
        Set<Integer> docIds = Arrays.stream(perLeafResult.getResult().scoreDocs).map(scoreDoc -> scoreDoc.doc).collect(Collectors.toSet());
        DocIdSetIterator allSiblings = queryUtils.getAllSiblings(
            leafReaderContext,
            docIds,
            knnQuery.getParentsFilter(),
            perLeafResult.getFilterBits()
        );

        final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
            .matchedDocsIterator(allSiblings)
            .numberOfMatchedDocs(allSiblings.cost())
            // setting to false because in re-scoring we want to do exact search on full precision vectors
            .useQuantizedVectorsForSearch(useQuantizedVectors)
            .k((int) allSiblings.cost())
            .field(knnQuery.getField())
            .radius(knnQuery.getRadius())
            .floatQueryVector(knnQuery.getQueryVector())
            .byteQueryVector(knnQuery.getByteQueryVector())
            .isMemoryOptimizedSearchEnabled(knnQuery.isMemoryOptimizedSearch())
            .build();
        TopDocs rescoreResult = knnWeight.exactSearch(leafReaderContext, exactSearcherContext);
        return new PerLeafResult(perLeafResult.getFilterBits(), rescoreResult);
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
                if (perLeafeResult.getResult().scoreDocs.length == 0) {
                    return perLeafeResult;
                }
                DocIdSetIterator matchedDocs = new TopDocsDISI(perLeafeResult.getResult());
                final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                        .matchedDocsIterator(matchedDocs)
                        .numberOfMatchedDocs(perLeafResults.get(finalI).getResult().scoreDocs.length)
                        // setting to false because in re-scoring we want to do exact search on full precision vectors
                        .useQuantizedVectorsForSearch(false)
                        .k(k)
                        .radius(knnQuery.getRadius())
                        .field(knnQuery.getField())
                        .floatQueryVector(knnQuery.getQueryVector())
                        .byteQueryVector(knnQuery.getByteQueryVector())
                        .isMemoryOptimizedSearchEnabled(knnQuery.isMemoryOptimizedSearch())
                        .build();
                TopDocs rescoreResult = knnWeight.exactSearch(leafReaderContext, exactSearcherContext);
                return new PerLeafResult(perLeafeResult.getFilterBits(), rescoreResult);
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks);
    }

    private PerLeafResult searchLeaf(LeafReaderContext ctx, KNNWeight queryWeight, int k) throws IOException {
        final PerLeafResult perLeafResult = queryWeight.searchLeaf(ctx, k);
        final Bits liveDocs = ctx.reader().getLiveDocs();
        if (liveDocs != null) {

            List<ScoreDoc> list = new ArrayList<>();
            for (ScoreDoc scoreDoc : perLeafResult.getResult().scoreDocs) {
                if (liveDocs.get(scoreDoc.doc)) {
                    list.add(scoreDoc);
                }
            }
            ScoreDoc[] filteredScoreDoc = list.toArray(new ScoreDoc[0]);
            TotalHits totalHits = new TotalHits(filteredScoreDoc.length, TotalHits.Relation.EQUAL_TO);
            return new PerLeafResult(perLeafResult.getFilterBits(), new TopDocs(totalHits, filteredScoreDoc));
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
