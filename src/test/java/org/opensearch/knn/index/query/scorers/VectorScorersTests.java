/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class VectorScorersTests extends KNNTestCase {

    private final FieldInfo fieldInfo = mock(FieldInfo.class);

    // ──────────────────────────────────────────────
    // Float target
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testFloatTarget_withBinaryDocValues_returnsKNNBinaryDocValuesScorer() {
        float[] query = { 1.0f, 2.0f, 3.0f };
        List<float[]> docs = List.of(new float[] { 1.0f, 2.0f, 3.0f }, new float[] { 4.0f, 5.0f, 6.0f });

        TestVectorValues.PredefinedFloatVectorBinaryDocValues binaryDocValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(
            docs
        );
        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(binaryDocValues);

        VectorScorer scorer = VectorScorers.createScorer(iteratorValues, query, VectorScorerMode.SCORE, SpaceType.L2, fieldInfo);

        assertNotNull(scorer);
        assertTrue(scorer instanceof KNNBinaryDocValuesScorer);

        assertScores(buildExpectedScores(query, docs, SpaceType.L2), scorer);
    }

    @SneakyThrows
    public void testFloatTarget_withFloatVectorValues_delegatesToScoreMode() {
        float[] query = { 1.0f, 2.0f };
        List<float[]> docs = List.of(new float[] { 1.0f, 2.0f }, new float[] { 3.0f, 4.0f });
        TestVectorValues.PreDefinedFloatVectorValues floatVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            docs,
            VectorSimilarityFunction.EUCLIDEAN
        );

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(floatVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(floatVectorValues);

        VectorScorer scorer = VectorScorers.createScorer(iteratorValues, query, VectorScorerMode.SCORE, SpaceType.L2, fieldInfo);

        assertNotNull(scorer);
        assertScores(buildExpectedScores(query, docs, SpaceType.L2), scorer);
    }

    @SneakyThrows
    public void testFloatTarget_withByteVectorValues_returnsADCScorer() {
        float[] query = { 1.0f, 2.0f };
        TestVectorValues.PreDefinedByteVectorValues byteVectorValues = new TestVectorValues.PreDefinedByteVectorValues(
            List.of(new byte[] { 1, 2 }, new byte[] { 3, 4 })
        );

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(byteVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(byteVectorValues);

        when(fieldInfo.getAttribute(QFRAMEWORK_CONFIG)).thenReturn("adc_config");
        when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.L2.getValue());
        QuantizationConfig adcConfig = QuantizationConfig.builder().enableADC(true).build();
        try (MockedStatic<QuantizationConfigParser> parserMock = mockStatic(QuantizationConfigParser.class)) {
            parserMock.when(() -> QuantizationConfigParser.fromCsv(anyString())).thenReturn(adcConfig);

            VectorScorer scorer = VectorScorers.createScorer(iteratorValues, query, VectorScorerMode.SCORE, SpaceType.L2, fieldInfo);

            assertNotNull(scorer);
            assertFalse(scorer instanceof KNNBinaryDocValuesScorer);
        }
    }

    @SneakyThrows
    public void testFloatTarget_withUnsupportedKnnVectorValues_throwsException() {
        float[] query = { 1.0f };
        KnnVectorValues unsupported = mock(KnnVectorValues.class);

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(mock(DocIdSetIterator.class));
        when(iteratorValues.getKnnVectorValues()).thenReturn(unsupported);

        try {
            VectorScorers.createScorer(iteratorValues, query, VectorScorerMode.SCORE, SpaceType.L2, fieldInfo);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Unsupported KnnVectorValues type"));
        }
    }

    // ──────────────────────────────────────────────
    // Byte target
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testByteTarget_withBinaryDocValues_returnsKNNBinaryDocValuesScorer() {
        byte[] query = { 0b0000_0000, (byte) 0b1111_1111 };
        List<byte[]> docs = List.of(new byte[] { 0b0000_0000, (byte) 0b1111_1111 }, new byte[] { (byte) 0b1111_1111, 0b0000_0000 });

        TestVectorValues.PredefinedByteVectorBinaryDocValues binaryDocValues = new TestVectorValues.PredefinedByteVectorBinaryDocValues(
            docs
        );
        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(binaryDocValues);

        VectorScorer scorer = VectorScorers.createScorer(iteratorValues, query, VectorScorerMode.SCORE, SpaceType.HAMMING, fieldInfo);

        assertNotNull(scorer);
        assertTrue(scorer instanceof KNNBinaryDocValuesScorer);

        assertScores(buildExpectedScores(query, docs, SpaceType.HAMMING), scorer);
    }

    @SneakyThrows
    public void testByteTarget_withByteVectorValues_delegatesToScoreMode() {
        byte[] query = { 1, 2 };
        List<byte[]> docs = List.of(new byte[] { 1, 2 }, new byte[] { 3, 4 });
        TestVectorValues.PreDefinedByteVectorValues byteVectorValues = new TestVectorValues.PreDefinedByteVectorValues(
            docs,
            VectorSimilarityFunction.EUCLIDEAN
        );

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(byteVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(byteVectorValues);

        VectorScorer scorer = VectorScorers.createScorer(iteratorValues, query, VectorScorerMode.SCORE, SpaceType.L2, fieldInfo);

        assertNotNull(scorer);
        assertScores(buildExpectedScores(query, docs, SpaceType.L2), scorer);
    }

    @SneakyThrows
    public void testByteTarget_withFloatVectorValues_throwsException() {
        byte[] query = { 1, 2 };
        FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(mock(DocIdSetIterator.class));
        when(iteratorValues.getKnnVectorValues()).thenReturn(floatVectorValues);

        try {
            VectorScorers.createScorer(iteratorValues, query, VectorScorerMode.SCORE, SpaceType.L2, fieldInfo);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("Byte target requires ByteVectorValues"));
        }
    }

    @SneakyThrows
    public void testByteTarget_withByteVectorValues_hammingSpaceType_returnsHammingScorer() {
        byte[] query = { 0b0000_0000, (byte) 0b1111_1111 };
        List<byte[]> docs = List.of(new byte[] { 0b0000_0000, (byte) 0b1111_1111 }, new byte[] { (byte) 0b1111_1111, 0b0000_0000 });
        TestVectorValues.PreDefinedByteVectorValues byteVectorValues = new TestVectorValues.PreDefinedByteVectorValues(docs);

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(byteVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(byteVectorValues);

        FieldInfo hammingFieldInfo = mock(FieldInfo.class);
        when(hammingFieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.HAMMING.getValue());

        VectorScorer scorer = VectorScorers.createScorer(
            iteratorValues,
            query,
            VectorScorerMode.SCORE,
            SpaceType.HAMMING,
            hammingFieldInfo
        );

        assertNotNull(scorer);
        assertFalse(scorer instanceof KNNBinaryDocValuesScorer);
        assertScores(buildExpectedScores(query, docs, SpaceType.HAMMING), scorer);
    }

    // ──────────────────────────────────────────────
    // Nested wrapping
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testFloatTarget_withParentBitSet_wrapsWithNestedScorer() {
        float[] query = { 1.0f, 2.0f };
        TestVectorValues.PreDefinedFloatVectorValues floatVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            List.of(new float[] { 1.0f, 2.0f }, new float[] { 3.0f, 4.0f })
        );

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(floatVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(floatVectorValues);

        VectorScorerMode vectorScorerMode = mock(VectorScorerMode.class);
        VectorScorer baseScorer = mock(VectorScorer.class);
        when(baseScorer.iterator()).thenReturn(mock(DocIdSetIterator.class));
        when(vectorScorerMode.createScorer(floatVectorValues, query)).thenReturn(baseScorer);

        BitSet parentBitSet = new FixedBitSet(4);
        parentBitSet.set(2);

        VectorScorer scorer = VectorScorers.createScorer(
            iteratorValues,
            query,
            vectorScorerMode,
            SpaceType.L2,
            fieldInfo,
            null,
            parentBitSet
        );

        assertTrue(scorer instanceof NestedBestChildVectorScorer);
    }

    @SneakyThrows
    public void testByteTarget_withParentBitSet_wrapsWithNestedScorer() {
        byte[] query = { 1, 2 };
        TestVectorValues.PreDefinedByteVectorValues byteVectorValues = new TestVectorValues.PreDefinedByteVectorValues(
            List.of(new byte[] { 1, 2 }, new byte[] { 3, 4 })
        );

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(byteVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(byteVectorValues);

        VectorScorerMode vectorScorerMode = mock(VectorScorerMode.class);
        VectorScorer baseScorer = mock(VectorScorer.class);
        when(baseScorer.iterator()).thenReturn(mock(DocIdSetIterator.class));
        when(vectorScorerMode.createScorer(byteVectorValues, query)).thenReturn(baseScorer);

        BitSet parentBitSet = new FixedBitSet(4);
        parentBitSet.set(2);

        VectorScorer scorer = VectorScorers.createScorer(
            iteratorValues,
            query,
            vectorScorerMode,
            SpaceType.L2,
            fieldInfo,
            null,
            parentBitSet
        );

        assertTrue(scorer instanceof NestedBestChildVectorScorer);
    }

    // ──────────────────────────────────────────────
    // CosineADCFlatVectorsScorer
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testFloatTarget_withByteVectorValues_cosineSpaceType_appliesCosineScoreConversion() {
        float[] query = { 1.0f, 2.0f };
        List<byte[]> docs = List.of(new byte[] { 1, 2 }, new byte[] { 3, 4 });
        TestVectorValues.PreDefinedByteVectorValues byteVectorValues = new TestVectorValues.PreDefinedByteVectorValues(docs);

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(byteVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(byteVectorValues);

        when(fieldInfo.getAttribute(QFRAMEWORK_CONFIG)).thenReturn("adc_config");
        when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.COSINESIMIL.getValue());
        QuantizationConfig adcConfig = QuantizationConfig.builder().enableADC(true).build();

        // Known scores the ADC scorer will return for each doc (now direct cosine-format scores)
        float[] scores = { 0.85f, 0.72f };

        // Build a mock FlatVectorsScorer that returns known scores
        RandomVectorScorer mockRandomScorer = mock(RandomVectorScorer.class);
        when(mockRandomScorer.score(0)).thenReturn(scores[0]);
        when(mockRandomScorer.score(1)).thenReturn(scores[1]);

        FlatVectorsScorer mockFlatScorer = mock(FlatVectorsScorer.class);
        when(mockFlatScorer.getRandomVectorScorer(any(VectorSimilarityFunction.class), any(KnnVectorValues.class), any(float[].class)))
            .thenReturn(mockRandomScorer);

        try (
            MockedStatic<QuantizationConfigParser> parserMock = mockStatic(QuantizationConfigParser.class);
            MockedStatic<FlatVectorsScorerProvider> providerMock = mockStatic(FlatVectorsScorerProvider.class)
        ) {
            parserMock.when(() -> QuantizationConfigParser.fromCsv(anyString())).thenReturn(adcConfig);
            providerMock.when(() -> FlatVectorsScorerProvider.getFlatVectorsScorer(any(), any(), any())).thenReturn(mockFlatScorer);

            VectorScorer scorer = VectorScorers.createScorer(
                iteratorValues,
                query,
                VectorScorerMode.SCORE,
                SpaceType.COSINESIMIL,
                fieldInfo
            );

            assertNotNull(scorer);

            // Scores pass through directly — no wrapper conversion
            Map<Integer, Float> expectedScores = new HashMap<>();
            for (int i = 0; i < scores.length; i++) {
                expectedScores.put(i, scores[i]);
            }
            assertScores(expectedScores, scorer);
        }
    }

    @SneakyThrows
    public void testFloatTarget_withByteVectorValues_cosineSpaceType_negativeIpScore() {
        float[] query = { 1.0f, 2.0f };
        List<byte[]> docs = List.of(new byte[] { 1, 2 });
        TestVectorValues.PreDefinedByteVectorValues byteVectorValues = new TestVectorValues.PreDefinedByteVectorValues(docs);

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(byteVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(byteVectorValues);

        when(fieldInfo.getAttribute(QFRAMEWORK_CONFIG)).thenReturn("adc_config");
        when(fieldInfo.getAttribute(SPACE_TYPE)).thenReturn(SpaceType.COSINESIMIL.getValue());
        QuantizationConfig adcConfig = QuantizationConfig.builder().enableADC(true).build();

        // Direct cosine-format score
        float score = 0.25f;
        RandomVectorScorer mockRandomScorer = mock(RandomVectorScorer.class);
        when(mockRandomScorer.score(0)).thenReturn(score);

        FlatVectorsScorer mockFlatScorer = mock(FlatVectorsScorer.class);
        when(mockFlatScorer.getRandomVectorScorer(any(VectorSimilarityFunction.class), any(KnnVectorValues.class), any(float[].class)))
            .thenReturn(mockRandomScorer);

        try (
            MockedStatic<QuantizationConfigParser> parserMock = mockStatic(QuantizationConfigParser.class);
            MockedStatic<FlatVectorsScorerProvider> providerMock = mockStatic(FlatVectorsScorerProvider.class)
        ) {
            parserMock.when(() -> QuantizationConfigParser.fromCsv(anyString())).thenReturn(adcConfig);
            providerMock.when(() -> FlatVectorsScorerProvider.getFlatVectorsScorer(any(), any(), any())).thenReturn(mockFlatScorer);

            VectorScorer scorer = VectorScorers.createScorer(
                iteratorValues,
                query,
                VectorScorerMode.SCORE,
                SpaceType.COSINESIMIL,
                fieldInfo
            );

            // Score passes through directly — no wrapper conversion
            assertScores(Map.of(0, score), scorer);
        }
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────

    private static Map<Integer, Float> buildExpectedScores(float[] query, List<float[]> docs, SpaceType spaceType) {
        return IntStream.range(0, docs.size())
            .boxed()
            .collect(Collectors.toMap(i -> i, i -> spaceType.getKnnVectorSimilarityFunction().compare(query, docs.get(i))));
    }

    private static Map<Integer, Float> buildExpectedScores(byte[] query, List<byte[]> docs, SpaceType spaceType) {
        return IntStream.range(0, docs.size())
            .boxed()
            .collect(Collectors.toMap(i -> i, i -> spaceType.getKnnVectorSimilarityFunction().compare(query, docs.get(i))));
    }

    private static void assertScores(Map<Integer, Float> expectedScores, VectorScorer scorer) throws IOException {
        Map<Integer, Float> actualScores = new HashMap<>();
        DocIdSetIterator iterator = scorer.iterator();
        for (int docId = iterator.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = iterator.nextDoc()) {
            actualScores.put(docId, scorer.score());
        }
        assertEquals(expectedScores, actualScores);
    }
}
