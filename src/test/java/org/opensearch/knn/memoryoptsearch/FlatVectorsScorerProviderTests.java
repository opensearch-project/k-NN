/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.index.engine.faiss.SQConfigParser;
import org.opensearch.knn.index.engine.faiss.SQConfig;
import org.opensearch.knn.memoryoptsearch.faiss.Faiss104ScalarQuantizedVectorScorer;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FlatVectorsScorerProviderTests extends KNNTestCase {
    private static final float[] FLOAT_QUERY = new float[] { 1.3f, 2.2f, -0.7f, 11.f };
    private static final float[] FLOAT_VECTOR = new float[] { 8.f, 3.7f, 4.12f, -0.3f };
    private static final byte[] BYTE_QUERY = new byte[] { 12, 77, 100, 4 };
    private static final byte[] BYTE_VECTOR = new byte[] { 1, 3, 0, 90 };

    private static final FlatVectorsScorer VECTOR_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();

    @SneakyThrows
    public void testAdcScoringL2() {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final String adcConfig = QuantizationConfigParser.toCsv(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).enableADC(true).build()
        );
        when(fieldInfo.getAttribute(KNNConstants.QFRAMEWORK_CONFIG)).thenReturn(adcConfig);
        when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());

        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            KNNVectorSimilarityFunction.EUCLIDEAN,
            VECTOR_SCORER
        );

        final ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        when(byteVectorValues.vectorValue(anyInt())).thenReturn(BYTE_VECTOR);
        final RandomVectorScorer vectorScorer = scorer.getRandomVectorScorer(null, byteVectorValues, FLOAT_QUERY);
        assertNotNull(vectorScorer);
        // Verify it scores without error
        vectorScorer.score(0);
    }

    @SneakyThrows
    public void testAdcScoringInnerProduct() {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final String adcConfig = QuantizationConfigParser.toCsv(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).enableADC(true).build()
        );
        when(fieldInfo.getAttribute(KNNConstants.QFRAMEWORK_CONFIG)).thenReturn(adcConfig);
        when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.INNER_PRODUCT.getValue());

        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            VECTOR_SCORER
        );

        final ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        when(byteVectorValues.vectorValue(anyInt())).thenReturn(BYTE_VECTOR);
        final RandomVectorScorer vectorScorer = scorer.getRandomVectorScorer(null, byteVectorValues, FLOAT_QUERY);
        assertNotNull(vectorScorer);
        vectorScorer.score(0);
    }

    @SneakyThrows
    public void testAdcScoringUnsupportedByteQuery() {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final String adcConfig = QuantizationConfigParser.toCsv(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).enableADC(true).build()
        );
        when(fieldInfo.getAttribute(KNNConstants.QFRAMEWORK_CONFIG)).thenReturn(adcConfig);
        when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());

        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            KNNVectorSimilarityFunction.EUCLIDEAN,
            VECTOR_SCORER
        );

        final ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        expectThrows(UnsupportedOperationException.class, () -> scorer.getRandomVectorScorer(null, byteVectorValues, BYTE_QUERY));
    }

    @SneakyThrows
    public void testFaissSQOneBitResolverReturnsFaissSQScorer() {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final String sqConfig = SQConfigParser.toCsv(SQConfig.builder().bits(FaissSQEncoder.Bits.ONE.getValue()).build());
        when(fieldInfo.getAttribute(KNNConstants.SQ_CONFIG)).thenReturn(sqConfig);

        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            KNNVectorSimilarityFunction.EUCLIDEAN,
            VECTOR_SCORER
        );

        assertNotSame("Expected a specialized SQ scorer, not the delegate", VECTOR_SCORER, scorer);
    }

    @SneakyThrows
    public void testFaissSQNonOneBitFallsBackToDelegate() {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        // bits=2 should NOT match the FaissSQScorerResolver
        final String sqConfig = SQConfigParser.toCsv(SQConfig.builder().bits(2).build());
        when(fieldInfo.getAttribute(KNNConstants.SQ_CONFIG)).thenReturn(sqConfig);

        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            KNNVectorSimilarityFunction.EUCLIDEAN,
            VECTOR_SCORER
        );

        assertSame("Expected delegate scorer for non-1-bit SQ field", VECTOR_SCORER, scorer);
    }

    @SneakyThrows
    public void testHammingScoring() {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.HAMMING.getValue());
        // Get hamming scorer
        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            KNNVectorSimilarityFunction.HAMMING,
            VECTOR_SCORER
        );

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

    public void testFaissSQScoring_whenSQFieldInfo_thenReturnsFaiss104ScalarQuantizedVectorScorer() {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getAttribute(KNNConstants.SQ_CONFIG)).thenReturn("bits=1");

        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            KNNVectorSimilarityFunction.EUCLIDEAN,
            VECTOR_SCORER
        );

        assertTrue(scorer instanceof Faiss104ScalarQuantizedVectorScorer);
    }

    public void testNonHammingScoring() {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        // Test L2
        when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
        doTest(KNNVectorSimilarityFunction.EUCLIDEAN, true, fieldInfo);
        doTest(KNNVectorSimilarityFunction.EUCLIDEAN, false, fieldInfo);

        // Test DotProduct
        doTest(KNNVectorSimilarityFunction.DOT_PRODUCT, true, fieldInfo);
        doTest(KNNVectorSimilarityFunction.DOT_PRODUCT, false, fieldInfo);

        // Test Cosine
        doTest(KNNVectorSimilarityFunction.COSINE, true, fieldInfo);
        doTest(KNNVectorSimilarityFunction.COSINE, false, fieldInfo);

        // Test Maximum Inner Product
        doTest(KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, true, fieldInfo);
        doTest(KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, false, fieldInfo);
    }

    @SneakyThrows
    private void doTest(final KNNVectorSimilarityFunction knnVectorSimilarityFunction, final boolean isFloat, final FieldInfo fieldInfo) {
        final FlatVectorsScorer scorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
            fieldInfo,
            knnVectorSimilarityFunction,
            VECTOR_SCORER
        );

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
