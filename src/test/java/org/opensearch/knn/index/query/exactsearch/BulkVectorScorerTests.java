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
import java.util.List;

public class BulkVectorScorerTests extends KNNTestCase {

    private static final float[] QUERY = new float[] { 1.0f, 0.0f, 0.0f };

    @SneakyThrows
    public void testForKSearch_euclidean_returnsAllDocsWithCorrectScores() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f }
        );

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            DocIdSetIterator.all(vectors.size())
        );
        DocIdSetIterator iter = scorer.iterator();

        for (int i = 0; i < vectors.size(); i++) {
            assertEquals(i, iter.nextDoc());
            assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(i)), scorer.score(), 1e-5f);
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testForKSearch_cosine_returnsAllDocsWithCorrectScores() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.707f, 0.707f, 0.0f },
            new float[] { -1.0f, 0.0f, 0.0f }
        );

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.COSINE),
            DocIdSetIterator.all(vectors.size())
        );
        DocIdSetIterator iter = scorer.iterator();

        for (int i = 0; i < vectors.size(); i++) {
            assertEquals(i, iter.nextDoc());
            assertEquals(VectorSimilarityFunction.COSINE.compare(QUERY, vectors.get(i)), scorer.score(), 1e-5f);
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testForKSearch_dotProduct_returnsAllDocsWithCorrectScores() {
        List<float[]> vectors = List.of(
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.3f, 0.3f, 0.3f }
        );

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.DOT_PRODUCT),
            DocIdSetIterator.all(vectors.size())
        );
        DocIdSetIterator iter = scorer.iterator();

        for (int i = 0; i < vectors.size(); i++) {
            assertEquals(i, iter.nextDoc());
            assertEquals(VectorSimilarityFunction.DOT_PRODUCT.compare(QUERY, vectors.get(i)), scorer.score(), 1e-5f);
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testForKSearch_filteredMatchedDocs_onlyScoresMatchedDocs() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.3f, 0.3f, 0.3f },
            new float[] { 0.9f, 0.1f, 0.0f }
        );

        FixedBitSet bitSet = new FixedBitSet(vectors.size());
        bitSet.set(1);
        bitSet.set(3);
        bitSet.set(4);

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            new BitSetIterator(bitSet, bitSet.cardinality())
        );
        DocIdSetIterator iter = scorer.iterator();

        assertEquals(1, iter.nextDoc());
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(1)), scorer.score(), 1e-5f);

        assertEquals(3, iter.nextDoc());
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(3)), scorer.score(), 1e-5f);

        assertEquals(4, iter.nextDoc());
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(4)), scorer.score(), 1e-5f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testForRadialSearch_euclidean_filtersOnBothMatchAndScore() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.9f, 0.1f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.8f, 0.2f, 0.0f }
        );

        float score0 = VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(0));
        float score1 = VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(1));
        float score2 = VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(2));
        float score3 = VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(3));
        float score4 = VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(4));

        float minScore = (score1 + score4) / 2.0f;
        assertTrue(score0 >= minScore);
        assertTrue(score1 < minScore);
        assertTrue(score2 >= minScore);
        assertTrue(score3 < minScore);
        assertTrue(score4 >= minScore);

        FixedBitSet bitSet = new FixedBitSet(vectors.size());
        bitSet.set(0);
        bitSet.set(2);
        bitSet.set(3);
        bitSet.set(4);

        BulkVectorScorer scorer = BulkVectorScorer.forRadialSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            new BitSetIterator(bitSet, bitSet.cardinality()),
            minScore
        );
        DocIdSetIterator iter = scorer.iterator();

        assertEquals(0, iter.nextDoc());
        assertEquals(score0, scorer.score(), 1e-5f);

        assertEquals(2, iter.nextDoc());
        assertEquals(score2, scorer.score(), 1e-5f);

        assertEquals(4, iter.nextDoc());
        assertEquals(score4, scorer.score(), 1e-5f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testForRadialSearch_cosine_filtersLowSimilarity() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.707f, 0.707f, 0.0f },
            new float[] { -1.0f, 0.0f, 0.0f }
        );

        float scoreDoc0 = VectorSimilarityFunction.COSINE.compare(QUERY, vectors.get(0));
        float scoreDoc2 = VectorSimilarityFunction.COSINE.compare(QUERY, vectors.get(2));
        float minScore = scoreDoc2 - 0.01f;

        BulkVectorScorer scorer = BulkVectorScorer.forRadialSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.COSINE),
            DocIdSetIterator.all(vectors.size()),
            minScore
        );
        DocIdSetIterator iter = scorer.iterator();

        assertEquals(0, iter.nextDoc());
        assertEquals(scoreDoc0, scorer.score(), 1e-5f);

        assertEquals(2, iter.nextDoc());
        assertEquals(scoreDoc2, scorer.score(), 1e-5f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testForRadialSearch_dotProduct_filtersLowScores() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f }
        );

        float scoreDoc0 = VectorSimilarityFunction.DOT_PRODUCT.compare(QUERY, vectors.get(0));
        float scoreDoc1 = VectorSimilarityFunction.DOT_PRODUCT.compare(QUERY, vectors.get(1));
        float scoreDoc2 = VectorSimilarityFunction.DOT_PRODUCT.compare(QUERY, vectors.get(2));
        float minScore = scoreDoc1 - 0.01f;

        assertTrue(scoreDoc0 >= minScore);
        assertTrue(scoreDoc1 >= minScore);
        assertTrue(scoreDoc2 < minScore);

        BulkVectorScorer scorer = BulkVectorScorer.forRadialSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.DOT_PRODUCT),
            DocIdSetIterator.all(vectors.size()),
            minScore
        );
        DocIdSetIterator iter = scorer.iterator();

        assertEquals(0, iter.nextDoc());
        assertEquals(scoreDoc0, scorer.score(), 1e-5f);

        assertEquals(1, iter.nextDoc());
        assertEquals(scoreDoc1, scorer.score(), 1e-5f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iter.nextDoc());
    }

    @SneakyThrows
    public void testForRadialSearch_noDocsPassFilter() {
        List<float[]> vectors = List.of(new float[] { 0.0f, 1.0f, 0.0f }, new float[] { 0.0f, 0.0f, 1.0f });

        BulkVectorScorer scorer = BulkVectorScorer.forRadialSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            DocIdSetIterator.all(vectors.size()),
            Float.MAX_VALUE
        );

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.iterator().nextDoc());
    }

    @SneakyThrows
    public void testIterator_advance_withFilteredDocs() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.3f, 0.3f, 0.3f },
            new float[] { 0.9f, 0.1f, 0.0f }
        );

        FixedBitSet bitSet = new FixedBitSet(vectors.size());
        bitSet.set(0);
        bitSet.set(2);
        bitSet.set(4);

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            new BitSetIterator(bitSet, bitSet.cardinality())
        );
        DocIdSetIterator iter = scorer.iterator();

        int doc = iter.advance(2);
        assertTrue(doc >= 2);
        assertNotEquals(DocIdSetIterator.NO_MORE_DOCS, doc);
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(doc)), scorer.score(), 1e-5f);
    }

    @SneakyThrows
    public void testIterator_advance_whenAlreadyAtOrPastTarget_returnsCurrentDoc() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.0f, 1.0f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.5f, 0.5f, 0.0f }
        );

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            DocIdSetIterator.all(vectors.size())
        );
        DocIdSetIterator iter = scorer.iterator();

        assertEquals(0, iter.nextDoc());
        assertEquals(1, iter.nextDoc());
        assertEquals(2, iter.nextDoc());

        assertEquals(2, iter.advance(2));
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(2)), scorer.score(), 1e-5f);

        assertEquals(2, iter.advance(1));
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(QUERY, vectors.get(2)), scorer.score(), 1e-5f);
    }

    @SneakyThrows
    public void testIterator_advancePastEnd() {
        List<float[]> vectors = List.of(new float[] { 1.0f, 0.0f, 0.0f });

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            DocIdSetIterator.all(vectors.size())
        );

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.iterator().advance(100));
    }

    @SneakyThrows
    public void testDocID_initialState() {
        List<float[]> vectors = List.of(new float[] { 1.0f, 0.0f, 0.0f });

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            DocIdSetIterator.all(vectors.size())
        );

        assertEquals(-1, scorer.docID());
    }

    @SneakyThrows
    public void testGetMaxScore_returnsMaxValue() {
        List<float[]> vectors = List.of(new float[] { 1.0f, 0.0f, 0.0f });

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            DocIdSetIterator.all(vectors.size())
        );

        assertEquals(Float.MAX_VALUE, scorer.getMaxScore(Integer.MAX_VALUE), 0.0f);
    }

    @SneakyThrows
    public void testIterator_cost_reflectsMatchedDocs() {
        List<float[]> vectors = List.of(
            new float[] { 1.0f, 0.0f, 0.0f },
            new float[] { 0.5f, 0.5f, 0.0f },
            new float[] { 0.0f, 0.0f, 1.0f },
            new float[] { 0.3f, 0.3f, 0.3f },
            new float[] { 0.9f, 0.1f, 0.0f }
        );

        FixedBitSet bitSet = new FixedBitSet(vectors.size());
        bitSet.set(1);
        bitSet.set(3);

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            new BitSetIterator(bitSet, bitSet.cardinality())
        );

        assertEquals(bitSet.cardinality(), scorer.iterator().cost());
    }

    @SneakyThrows
    public void testEmptyMatchedDocs() {
        List<float[]> vectors = List.of(new float[] { 1.0f, 0.0f, 0.0f });

        BulkVectorScorer scorer = BulkVectorScorer.forKSearch(
            createVectorScorer(vectors, QUERY, VectorSimilarityFunction.EUCLIDEAN),
            DocIdSetIterator.empty()
        );

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.iterator().nextDoc());
    }

    private VectorScorer createVectorScorer(List<float[]> vectors, float[] query, VectorSimilarityFunction similarity) throws IOException {
        TestVectorValues.PreDefinedFloatVectorValues floatVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectors,
            similarity
        );
        return floatVectorValues.scorer(query);
    }
}
