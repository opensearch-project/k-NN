/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.mockito.MockedStatic;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.util.List;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class VectorScorersTests extends TestCase {

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

        DocIdSetIterator iterator = scorer.iterator();
        assertEquals(0, iterator.nextDoc());
        float scoreIdentical = scorer.score();

        assertEquals(1, iterator.nextDoc());
        float scoreFar = scorer.score();

        assertTrue("Identical vector should score higher", scoreIdentical > scoreFar);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testFloatTarget_withFloatVectorValues_delegatesToScoreMode() {
        float[] query = { 1.0f, 2.0f };
        TestVectorValues.PreDefinedFloatVectorValues floatVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            List.of(new float[] { 1.0f, 2.0f }, new float[] { 3.0f, 4.0f })
        );

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(floatVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(floatVectorValues);

        VectorScorerMode vectorScorerMode = mock(VectorScorerMode.class);
        VectorScorer expectedScorer = mock(VectorScorer.class);
        when(vectorScorerMode.createScorer(floatVectorValues, query)).thenReturn(expectedScorer);

        VectorScorer scorer = VectorScorers.createScorer(iteratorValues, query, vectorScorerMode, SpaceType.L2, fieldInfo);

        assertSame(expectedScorer, scorer);
        verify(vectorScorerMode).createScorer(floatVectorValues, query);
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

        DocIdSetIterator iterator = scorer.iterator();
        assertEquals(0, iterator.nextDoc());
        float scoreIdentical = scorer.score();

        assertEquals(1, iterator.nextDoc());
        float scoreFlipped = scorer.score();

        assertTrue("Identical vector should score higher", scoreIdentical > scoreFlipped);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testByteTarget_withByteVectorValues_delegatesToScoreMode() {
        byte[] query = { 1, 2 };
        TestVectorValues.PreDefinedByteVectorValues byteVectorValues = new TestVectorValues.PreDefinedByteVectorValues(
            List.of(new byte[] { 1, 2 }, new byte[] { 3, 4 })
        );

        KNNVectorValuesIterator.DocIdsIteratorValues iteratorValues = mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        when(iteratorValues.getDocIdSetIterator()).thenReturn(byteVectorValues.iterator());
        when(iteratorValues.getKnnVectorValues()).thenReturn(byteVectorValues);

        VectorScorerMode vectorScorerMode = mock(VectorScorerMode.class);
        VectorScorer expectedScorer = mock(VectorScorer.class);
        when(vectorScorerMode.createScorer(byteVectorValues, query)).thenReturn(expectedScorer);

        VectorScorer scorer = VectorScorers.createScorer(iteratorValues, query, vectorScorerMode, SpaceType.L2, fieldInfo);

        assertSame(expectedScorer, scorer);
        verify(vectorScorerMode).createScorer(byteVectorValues, query);
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
}
