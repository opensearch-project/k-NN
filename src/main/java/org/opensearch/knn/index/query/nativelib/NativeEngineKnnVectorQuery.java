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
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;

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

        List<Callable<TopDocs>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            tasks.add(() -> searchLeaf(leafReaderContext, knnWeight));
        }
        TopDocs[] perLeafResults = indexSearcher.getTaskExecutor().invokeAll(tasks).toArray(TopDocs[]::new);
        // TopDocs.merge requires perLeafResults to be sorted in descending order.
        TopDocs topK = TopDocs.merge(knnQuery.getK(), perLeafResults);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery();
        }
        return createRewrittenQuery(reader, topK);
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

    private static int[] findSegmentStarts(IndexReader reader, int[] docs) {
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

    private TopDocs searchLeaf(LeafReaderContext ctx, KNNWeight queryWeight) throws IOException {
        int totalHits = 0;
        final Map<Integer, Float> leafDocScores = queryWeight.searchLeaf(ctx);
        final List<ScoreDoc> scoreDocs = new ArrayList<>();
        final Bits liveDocs = ctx.reader().getLiveDocs();

        if (!leafDocScores.isEmpty()) {
            final List<Map.Entry<Integer, Float>> topScores = new ArrayList<>(leafDocScores.entrySet());
            topScores.sort(Map.Entry.<Integer, Float>comparingByValue().reversed());

            for (Map.Entry<Integer, Float> entry : topScores) {
                if (liveDocs == null || liveDocs.get(entry.getKey())) {
                    ScoreDoc scoreDoc = new ScoreDoc(entry.getKey() + ctx.docBase, entry.getValue());
                    scoreDocs.add(scoreDoc);
                    totalHits++;
                }
            }
        }

        return new TopDocs(new TotalHits(totalHits, TotalHits.Relation.EQUAL_TO), scoreDocs.toArray(ScoreDoc[]::new));
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
