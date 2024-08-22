/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.nativelib;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;
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
@Getter
@RequiredArgsConstructor
public class NativeEngineKnnVectorQuery extends Query {

    private final KNNQuery knnQuery;

    @Override
    public Query rewrite(final IndexSearcher indexSearcher) throws IOException {
        final IndexReader reader = indexSearcher.getIndexReader();
        final KNNWeight knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, ScoreMode.COMPLETE, 1);
        List<LeafReaderContext> leafReaderContexts = reader.leaves();

        List<Map<Integer, Float>> perLeafResults;
        RescoreContext rescoreContext = knnQuery.getRescoreContext();
        int finalK = knnQuery.getK();
        if (rescoreContext == null) {
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, finalK);
        } else {
            int firstPassK = rescoreContext.getFirstPassK(finalK);
            perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, firstPassK);
            ResultUtil.reduceToTopK(perLeafResults, firstPassK);
            perLeafResults = doRescore(indexSearcher, leafReaderContexts, knnWeight, perLeafResults, finalK);
        }
        ResultUtil.reduceToTopK(perLeafResults, finalK);
        TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        for (int i = 0; i < perLeafResults.size(); i++) {
            topDocs[i] = ResultUtil.resultMapToTopDocs(perLeafResults.get(i), leafReaderContexts.get(i).docBase);
        }

        TopDocs topK = TopDocs.merge(knnQuery.getK(), topDocs);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery();
        }
        return createRewrittenQuery(reader, topK);
    }

    private List<Map<Integer, Float>> doSearch(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        int k
    ) throws IOException {
        List<Callable<Map<Integer, Float>>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            tasks.add(() -> searchLeaf(leafReaderContext, knnWeight, k));
        }
        return indexSearcher.getTaskExecutor().invokeAll(tasks);
    }

    private List<Map<Integer, Float>> doRescore(
        final IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        List<Map<Integer, Float>> perLeafResults,
        int k
    ) throws IOException {
        List<Callable<Map<Integer, Float>>> rescoreTasks = new ArrayList<>(leafReaderContexts.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            int finalI = i;
            rescoreTasks.add(() -> {
                BitSet convertedBitSet = ResultUtil.resultMapToMatchBitSet(perLeafResults.get(finalI));
                return knnWeight.exactSearch(leafReaderContext, convertedBitSet, false, k);
            });
        }
        return indexSearcher.getTaskExecutor().invokeAll(rescoreTasks);
    }

    private Query createRewrittenQuery(IndexReader reader, TopDocs topK) {
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

    private Map<Integer, Float> searchLeaf(LeafReaderContext ctx, KNNWeight queryWeight, int k) throws IOException {
        final Map<Integer, Float> leafDocScores = queryWeight.searchLeaf(ctx, k);
        final Bits liveDocs = ctx.reader().getLiveDocs();
        if (liveDocs != null) {
            leafDocScores.entrySet().removeIf(entry -> liveDocs.get(entry.getKey()) == false);
        }
        return leafDocScores;
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
