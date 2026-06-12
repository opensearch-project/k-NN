/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BulkVectorScorerTests extends KNNTestCase {

    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;

    @SneakyThrows
    public void testFullPrecision_allDocsMatched_returnsCorrectScores() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f }
        );
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        DocIdSetIterator matchedDocs = DocIdSetIterator.all(vectors.size());
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs);

        DocIdSetIterator iter = scorer.iterator();

        assertEquals(0, iter.nextDoc());
        assertEquals(SIMILARITY.compare(query, vectors.get(0)), scorer.score(), 1e-5f);

        assertEquals(1, iter.nextDoc());
        assertEquals(SIMILARITY.compare(query, vectors.get(1)), scorer.score(), 1e-5f);

        assertEquals(2, iter.nextDoc());
        assertEquals(SIMILARITY.compare(query, vectors.get(2)), scorer.score(), 1e-5f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testFullPrecision_filteredMatchedDocs_onlyScoresMatchedDocs() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.3f, 0.3f, 0.3f },
            new float[] { 0.9f, 0.1f, 0.0f }
        );
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        // Only match docs 1, 3, 4 — skip docs 0 and 2
        FixedBitSet bitSet = new FixedBitSet(vectors.size());
        bitSet.set(1);
        bitSet.set(3);
        bitSet.set(4);
        DocIdSetIterator matchedDocs = new BitSetIterator(bitSet, bitSet.cardinality());

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs);

        DocIdSetIterator iter = scorer.iterator();

        assertEquals(1, iter.nextDoc());
        assertEquals(SIMILARITY.compare(query, vectors.get(1)), scorer.score(), 1e-5f);

        assertEquals(3, iter.nextDoc());
        assertEquals(SIMILARITY.compare(query, vectors.get(3)), scorer.score(), 1e-5f);

        assertEquals(4, iter.nextDoc());
        assertEquals(SIMILARITY.compare(query, vectors.get(4)), scorer.score(), 1e-5f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testMinScore_filteredMatchedDocs_filtersOnBothMatchAndScore() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.9f, 0.1f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.8f, 0.2f, 0.0f }
        );
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        float score0 = SIMILARITY.compare(query, vectors.get(0));
        float score1 = SIMILARITY.compare(query, vectors.get(1));
        float score2 = SIMILARITY.compare(query, vectors.get(2));
        float score3 = SIMILARITY.compare(query, vectors.get(3));
        float score4 = SIMILARITY.compare(query, vectors.get(4));

        // Choose minScore so that docs 1 and 3 (orthogonal vectors) are filtered by score
        float minScore = (score1 + score4) / 2.0f;
        assertTrue("doc 0 should pass score filter", score0 >= minScore);
        assertTrue("doc 1 should fail score filter", score1 < minScore);
        assertTrue("doc 2 should pass score filter", score2 >= minScore);
        assertTrue("doc 3 should fail score filter", score3 < minScore);
        assertTrue("doc 4 should pass score filter", score4 >= minScore);

        // Only match docs 0, 2, 3, 4 — doc 0 is NOT in matched set
        FixedBitSet bitSet = new FixedBitSet(vectors.size());
        bitSet.set(0);
        bitSet.set(2);
        bitSet.set(3);
        bitSet.set(4);
        DocIdSetIterator matchedDocs = new BitSetIterator(bitSet, bitSet.cardinality());

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs, minScore);

        DocIdSetIterator iter = scorer.iterator();

        // doc 0: in matched set AND passes score filter
        assertEquals(0, iter.nextDoc());
        assertEquals(score0, scorer.score(), 1e-5f);

        // doc 1: NOT in matched set — skipped
        // doc 2: in matched set AND passes score filter
        assertEquals(2, iter.nextDoc());
        assertEquals(score2, scorer.score(), 1e-5f);

        // doc 3: in matched set but fails score filter — skipped
        // doc 4: in matched set AND passes score filter
        assertEquals(4, iter.nextDoc());
        assertEquals(score4, scorer.score(), 1e-5f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testMinScore_noDocsPassFilter() {
        List<float[]> vectors = List.of(
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f }
        );
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        DocIdSetIterator matchedDocs = DocIdSetIterator.all(vectors.size());
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs, Float.MAX_VALUE);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.iterator().nextDoc());
    }

    @SneakyThrows
    public void testIterator_advance_withFilteredDocs() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.3f, 0.3f, 0.3f }
        );
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        FixedBitSet bitSet = new FixedBitSet(vectors.size());
        bitSet.set(0);
        bitSet.set(2);
        bitSet.set(4);
        DocIdSetIterator matchedDocs = new BitSetIterator(bitSet, bitSet.cardinality());

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs);

        DocIdSetIterator iter = scorer.iterator();
        // Advance past doc 0 and 1 — should land on doc 2 (next matched doc >= 2)
        int doc = iter.advance(2);
        assertTrue(doc >= 2);
        assertNotEquals(DocIdSetIterator.NO_MORE_DOCS, doc);
        assertEquals(SIMILARITY.compare(query, vectors.get(doc)), scorer.score(), 1e-5f);
    }

    @SneakyThrows
    public void testIterator_advancePastEnd() {
        List<float[]> vectors = List.of(new float[] { 1.0f, 0.0f, 0.0f });
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        DocIdSetIterator matchedDocs = DocIdSetIterator.all(vectors.size());
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.iterator().advance(100));
    }

    @SneakyThrows
    public void testDocID_initialState() {
        List<float[]> vectors = List.of(new float[] { 1.0f, 0.0f, 0.0f });
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        DocIdSetIterator matchedDocs = DocIdSetIterator.all(vectors.size());
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs);

        assertEquals(-1, scorer.docID());
    }

    @SneakyThrows
    public void testGetMaxScore_returnsMaxValue() {
        List<float[]> vectors = List.of(new float[] { 1.0f, 0.0f, 0.0f });
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        DocIdSetIterator matchedDocs = DocIdSetIterator.all(vectors.size());
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs);

        assertEquals(Float.MAX_VALUE, scorer.getMaxScore(Integer.MAX_VALUE), 0.0f);
    }

    @SneakyThrows
    public void testIterator_cost_reflectsMatchedDocs() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.3f, 0.3f, 0.3f }
        );
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        FixedBitSet bitSet = new FixedBitSet(vectors.size());
        bitSet.set(1);
        bitSet.set(3);
        DocIdSetIterator matchedDocs = new BitSetIterator(bitSet, bitSet.cardinality());

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs);

        assertEquals(bitSet.cardinality(), scorer.iterator().cost());
    }

    @SneakyThrows
    public void testEmptyMatchedDocs() {
        List<float[]> vectors = List.of(new float[] { 1.0f, 0.0f, 0.0f });
        float[] query = new float[] { 1.0f, 0.0f, 0.0f };

        VectorScorer vectorScorer = createVectorScorer(vectors, query);
        DocIdSetIterator matchedDocs = DocIdSetIterator.empty();
        BulkVectorScorer scorer = BulkVectorScorer.fullPrecision(vectorScorer, matchedDocs);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.iterator().nextDoc());
    }

    private VectorScorer createVectorScorer(List<float[]> vectors, float[] query) throws IOException {
        TestVectorValues.PreDefinedFloatVectorValues floatVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectors,
            SIMILARITY
        );
        return floatVectorValues.scorer(query);
    }
}
