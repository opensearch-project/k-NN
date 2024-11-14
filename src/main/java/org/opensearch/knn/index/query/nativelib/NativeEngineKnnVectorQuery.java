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
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
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

    @Override
    public Weight createWeight(IndexSearcher indexSearcher, ScoreMode scoreMode, float boost) throws IOException {
        final IndexReader reader = indexSearcher.getIndexReader();
        final KNNWeight knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, scoreMode, 1);
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        List<Map.Entry<LeafReaderContext, Map<Integer, Float>>> perLeafResults;
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
            perLeafResults = doRescore(indexSearcher, knnWeight, perLeafResults, finalK);
            long rescoreTime = stopWatch.stop().totalTime().millis();
            log.debug("Rescoring results took {} ms. oversampled k:{}, segments:{}", rescoreTime, firstPassK, leafReaderContexts.size());
        }
        ResultUtil.reduceToTopK(perLeafResults, finalK);
        TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        int i = 0;
        for (Map.Entry<LeafReaderContext, Map<Integer, Float>> entry : perLeafResults) {
            topDocs[i++] = ResultUtil.resultMapToTopDocs(entry.getValue(), entry.getKey().docBase);
        }

        TopDocs topK = TopDocs.merge(knnQuery.getK(), topDocs);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(indexSearcher, scoreMode, boost);
        }
        return createDocAndScoreQuery(reader, topK).createWeight(indexSearcher, scoreMode, boost);
    }

    private List<Map.Entry<LeafReaderContext, Map<Integer, Float>>> doSearch(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        int k
    ) throws IOException {
        List<Callable<Map.Entry<LeafReaderContext, Map<Integer, Float>>>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            tasks.add(() -> searchLeaf(leafReaderContext, knnWeight, k));
        }
        return indexSearcher.getTaskExecutor().invokeAll(tasks);
    }

    private List<Map.Entry<LeafReaderContext, Map<Integer, Float>>> doRescore(
        final IndexSearcher indexSearcher,
        KNNWeight knnWeight,
        List<Map.Entry<LeafReaderContext, Map<Integer, Float>>> perLeafResults,
        int k
    ) throws IOException {
        List<Callable<Map.Entry<LeafReaderContext, Map<Integer, Float>>>> rescoreTasks = new ArrayList<>(perLeafResults.size());
        for (Map.Entry<LeafReaderContext, Map<Integer, Float>> entry : perLeafResults) {
            rescoreTasks.add(() -> {
                BitSet convertedBitSet = ResultUtil.resultMapToMatchBitSet(entry.getValue());
                final ExactSearcher.ExactSearcherContext exactSearcherContext = ExactSearcher.ExactSearcherContext.builder()
                    .matchedDocs(convertedBitSet)
                    // setting to false because in re-scoring we want to do exact search on full precision vectors
                    .useQuantizedVectorsForSearch(false)
                    .k(k)
                    .isParentHits(false)
                    .knnQuery(knnQuery)
                    .build();
                final Map<Integer, Float> docIdScoreMap = knnWeight.exactSearch(entry.getKey(), exactSearcherContext);
                return new AbstractMap.SimpleEntry<>(entry.getKey(), docIdScoreMap);
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks);
    }

    private Query createDocAndScoreQuery(IndexReader reader, TopDocs topK) {
        int len = topK.scoreDocs.length;
        Arrays.sort(topK.scoreDocs, Comparator.comparingInt(a -> a.doc));
        int[] docs = new int[len];
        float[] scores = new float[len];
        for (int i = 0; i < len; i++) {
            docs[i] = topK.scoreDocs[i].doc;
            scores[i] = topK.scoreDocs[i].score;
        }
        int[] segmentStarts = findSegmentStarts(reader, docs);
        return new DocAndScoreQuery(knnQuery.getK(), docs, scores, segmentStarts, reader.getContext().id());
    }

    static int[] findSegmentStarts(IndexReader reader, int[] docs) {
        int[] starts = new int[reader.leaves().size() + 1];
        starts[starts.length - 1] = docs.length;
        if (starts.length == 2) {
            return starts;
        }
        int resultIndex = 0;
        for (int i = 1; i < starts.length - 1; i++) {
            int upper = reader.leaves().get(i).docBase;
            resultIndex = Arrays.binarySearch(docs, resultIndex, docs.length, upper);
            if (resultIndex < 0) {
                resultIndex = -1 - resultIndex;
            }
            starts[i] = resultIndex;
        }
        return starts;
    }

    private Map.Entry<LeafReaderContext, Map<Integer, Float>> searchLeaf(LeafReaderContext ctx, KNNWeight queryWeight, int k)
        throws IOException {
        final Map<Integer, Float> leafDocScores = queryWeight.searchLeaf(ctx, k);
        final Bits liveDocs = ctx.reader().getLiveDocs();
        if (liveDocs != null) {
            leafDocScores.entrySet().removeIf(entry -> liveDocs.get(entry.getKey()) == false);
        }
        return new AbstractMap.SimpleEntry<>(ctx, leafDocScores);
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
