/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FlatVectorsScorerProviderTests extends KNNTestCase {
    private static final float[] FLOAT_QUERY = new float[] { 1.3f, 2.2f, -0.7f, 11.f };
    private static final float[] FLOAT_VECTOR = new float[] { 8.f, 3.7f, 4.12f, -0.3f };
    private static final byte[] BYTE_QUERY = new byte[] { 12, 77, 100, 4 };
    private static final byte[] BYTE_VECTOR = new byte[] { 1, 3, 0, 90 };

    @SneakyThrows
    public void testHammingScoring() {
        // Get hamming scorer
        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(KNNVectorSimilarityFunction.HAMMING);

        // Test byte[] vector and query
        final ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        when(byteVectorValues.vectorValue(anyInt())).thenReturn(BYTE_VECTOR);
        final RandomVectorScorer vectorScorer = scorer.getRandomVectorScorer(null, byteVectorValues, BYTE_QUERY);

        // Validate score
        final float score = vectorScorer.score(0);
        final float expectedScore = KNNVectorSimilarityFunction.HAMMING.compare(BYTE_VECTOR, BYTE_QUERY);
        assertEquals(expectedScore, score, 1e-6);

        // Test float[] vector and query, it is not supported
        try {
            scorer.getRandomVectorScorer(null, byteVectorValues, new float[] {});
            fail();
        } catch (UnsupportedOperationException e) {
            // Expected
        }

        // Test non-ByteVectorValues is not supported
        final FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        try {
            scorer.getRandomVectorScorer(null, floatVectorValues, BYTE_QUERY);
            fail();
        } catch (IllegalArgumentException e) {
            // Expected
        }
    }

    public void testNonHammingScoring() {
        // Test L2
        doTest(KNNVectorSimilarityFunction.EUCLIDEAN, true);
        doTest(KNNVectorSimilarityFunction.EUCLIDEAN, false);

        // Test DotProduct
        doTest(KNNVectorSimilarityFunction.DOT_PRODUCT, true);
        doTest(KNNVectorSimilarityFunction.DOT_PRODUCT, false);

        // Test Cosine
        doTest(KNNVectorSimilarityFunction.COSINE, true);
        doTest(KNNVectorSimilarityFunction.COSINE, false);

        // Test Maximum Inner Product
        doTest(KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, true);
        doTest(KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, false);
    }

    @SneakyThrows
    private void doTest(final KNNVectorSimilarityFunction knnVectorSimilarityFunction, final boolean isFloat) {
        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(knnVectorSimilarityFunction);

        if (isFloat) {
            final FloatVectorValues vectorValues = mock(FloatVectorValues.class);
            when(vectorValues.vectorValue(anyInt())).thenReturn(FLOAT_VECTOR);
            when(vectorValues.dimension()).thenReturn(FLOAT_VECTOR.length);
            final RandomVectorScorer vectorScorer = scorer.getRandomVectorScorer(
                knnVectorSimilarityFunction.getVectorSimilarityFunction(),
                vectorValues,
                FLOAT_QUERY
            );
            final float score = vectorScorer.score(0);
            final float expected = knnVectorSimilarityFunction.compare(FLOAT_VECTOR, FLOAT_QUERY);
            assertEquals(expected, score, 1e-6);
        } else {
            final ByteVectorValues vectorValues = mock(ByteVectorValues.class);
            when(vectorValues.vectorValue(anyInt())).thenReturn(BYTE_VECTOR);
            when(vectorValues.dimension()).thenReturn(BYTE_VECTOR.length);
            final RandomVectorScorer vectorScorer = scorer.getRandomVectorScorer(
                knnVectorSimilarityFunction.getVectorSimilarityFunction(),
                vectorValues,
                BYTE_QUERY
            );
            final float score = vectorScorer.score(0);
            final float expected = knnVectorSimilarityFunction.compare(BYTE_VECTOR, BYTE_QUERY);
            assertEquals(expected, score, 1e-6);
        }
    }
}
