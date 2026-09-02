/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;
import org.opensearch.knn.memoryoptsearch.faiss.cagra.FaissHNSWCagraIndex;

import java.lang.reflect.Field;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.SQ_CONFIG;

public class FaissFlatIndexFactoryTests extends KNNTestCase {

    @SneakyThrows
    public void testCreate_whenSQOneBitField_thenReturnsFaissScalarQuantizedFlatIndex() {
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").addAttribute(SQ_CONFIG, "bits=1").build();
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissBinaryIndex result = FaissFlatIndexFactory.createBinaryIndex(fieldInfo, mockReader);

        assertNotNull(result);
        assertTrue(result instanceof FaissScalarQuantizedFlatIndex);
    }

    @SneakyThrows
    public void testCreate_whenNonSQField_thenReturnsNull() {
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissBinaryIndex result = FaissFlatIndexFactory.createBinaryIndex(fieldInfo, mockReader);

        assertNull(result);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenEmptyFlatVectors_thenSetsFlatBinaryIndex() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        final FaissHNSW faissHNSW = mock(FaissHNSW.class);
        FaissBinaryHnswIndex hnswIndex = new FaissBinaryHnswIndex(FaissBinaryHnswIndex.IBHF, faissHNSW);
        hnswIndex.setStorage(null);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");
        when(fieldInfo.getAttribute(SQ_CONFIG)).thenReturn("bits=1");

        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mockReader);

        assertNotNull(hnswIndex.getStorage());
        assertTrue(hnswIndex.getStorage() instanceof FaissScalarQuantizedFlatIndex);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenFlatVectorsNotEmptyBinaryIndex_thenDoesNotOverwrite() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        final FaissHNSW faissHNSW = mock(FaissHNSW.class);
        FaissBinaryHnswIndex hnswIndex = new FaissBinaryHnswIndex(FaissBinaryHnswIndex.IBHF, faissHNSW);
        final FaissBinaryHnswIndex existing = mock(FaissBinaryHnswIndex.class);
        hnswIndex.setStorage(existing);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mockReader);

        assertSame(existing, hnswIndex.getStorage());
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenNotIdMapBinaryIndex_thenNoOp() {
        FaissIndex nonIdMapIndex = mock(FaissIndex.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        // Should complete without throwing
        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(nonIdMapIndex, fieldInfo, mock(FlatVectorsReader.class));
    }

    @SneakyThrows
    public void testMaybeSetFlatBinaryIndex_whenNestedIsNotHNSW_thenNoOp() {
        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(mock(FaissIndex.class));

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        // Should complete without throwing
        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mock(FlatVectorsReader.class));
    }

    @SneakyThrows
    public void testMaybeSetFlatBinaryIndex_whenUnsupportedMetricType_thenThrowsIllegalArgument() {
        final FaissHNSW faissHNSW = mock(FaissHNSW.class);
        FaissBinaryHnswIndex hnswIndex = new FaissBinaryHnswIndex(FaissBinaryHnswIndex.IBHF, faissHNSW);
        hnswIndex.setStorage(null);

        FaissIdMapIndex idMapIndex = new FaissIdMapIndex(FaissIdMapIndex.IXMP);
        Field nestedIndexField = FaissIdMapIndex.class.getDeclaredField("nestedIndex");
        nestedIndexField.setAccessible(true);
        nestedIndexField.set(idMapIndex, hnswIndex);

        // Set metricType to an out-of-bounds value via reflection on FaissBinaryIndex
        Field metricTypeField = FaissBinaryIndex.class.getDeclaredField("metricType");
        metricTypeField.setAccessible(true);
        metricTypeField.setInt(idMapIndex, 99);

        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").addAttribute(SQ_CONFIG, "bits=1").build();

        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mock(FlatVectorsReader.class))
        );
        assertTrue(e.getMessage().contains("Unsupported metric type"));
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenFlatBinaryIndexFactoryReturnsNull_thenThrowsIllegalState() {
        final FaissHNSW faissHNSW = mock(FaissHNSW.class);
        FaissBinaryHnswIndex hnswIndex = new FaissBinaryHnswIndex(FaissBinaryHnswIndex.IBHF, faissHNSW);
        hnswIndex.setStorage(null);

        FaissIdMapIndex idMapIndex = new FaissIdMapIndex(FaissIdMapIndex.IXMP);
        Field nestedIndexField = FaissIdMapIndex.class.getDeclaredField("nestedIndex");
        nestedIndexField.setAccessible(true);
        nestedIndexField.set(idMapIndex, hnswIndex);

        FieldInfo nonSQFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();

        try {
            FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, nonSQFieldInfo, mock(FlatVectorsReader.class));
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains(FaissEmptyIndex.class.getName()));
        }
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenCagraWithEmptyFlatVectors_thenSetsFlatIndex() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissHNSWCagraIndex cagraIndex = new FaissHNSWCagraIndex(FaissHNSWCagraIndex.IHNC);
        cagraIndex.setFlatVectors(FaissEmptyIndex.INSTANCE);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(cagraIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");
        when(fieldInfo.getAttribute(SQ_CONFIG)).thenReturn("bits=1");

        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mockReader);

        assertNotNull(cagraIndex.getFlatVectors());
        assertFalse(FaissEmptyIndex.isEmptyIndex(cagraIndex.getFlatVectors()));
        assertTrue(cagraIndex.getFlatVectors() instanceof FaissScalarQuantizedFlatIndex);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenCagraWithExistingFlatVectors_thenNoOp() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissHNSWCagraIndex cagraIndex = new FaissHNSWCagraIndex(FaissHNSWCagraIndex.IHNC);
        FaissIndex existingFlat = mock(FaissIndex.class);
        cagraIndex.setFlatVectors(existingFlat);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(cagraIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mockReader);

        assertSame(existingFlat, cagraIndex.getFlatVectors());
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenCagraWithEmptyFlatVectorsAndFP32Field_thenSetsFP32FlatIndex() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissHNSWCagraIndex cagraIndex = new FaissHNSWCagraIndex(FaissHNSWCagraIndex.IHNC);
        cagraIndex.setFlatVectors(FaissEmptyIndex.INSTANCE);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(cagraIndex);

        // Default field is full-precision FP32 (non-SQ). With flat-vector dedup, a graph-only float CAGRA index is now
        // served from Lucene's .vec via FaissFP32FlatIndex instead of throwing (previously unsupported).
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();

        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mockReader);

        assertNotNull(cagraIndex.getFlatVectors());
        assertFalse(FaissEmptyIndex.isEmptyIndex(cagraIndex.getFlatVectors()));
        assertTrue(cagraIndex.getFlatVectors() instanceof FaissFP32FlatIndex);
    }

    @SneakyThrows
    public void testCreate_whenFP32Field_thenReturnsFaissFP32FlatIndex() {
        // A default (non-SQ, full-precision) float field maps to the FP32 flat index.
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissIndex result = FaissFlatIndexFactory.createFlatIndex(fieldInfo, mockReader);

        assertTrue(result instanceof FaissFP32FlatIndex);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenFloatHnswWithEmptyFlatVectors_thenSetsFP32FlatIndex() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        // Plain float HNSW (IHNf) whose flat storage was skipped (IO_FLAG_SKIP_STORAGE) loads a FaissEmptyIndex.
        FaissHNSWIndex hnswIndex = new FaissHNSWIndex(FaissHNSWIndex.IHNF);
        hnswIndex.setFlatVectors(FaissEmptyIndex.INSTANCE);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();

        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mockReader);

        assertNotNull(hnswIndex.getFlatVectors());
        assertFalse(FaissEmptyIndex.isEmptyIndex(hnswIndex.getFlatVectors()));
        assertTrue(hnswIndex.getFlatVectors() instanceof FaissFP32FlatIndex);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenFloatHnswWithExistingFlatVectors_thenNoOp() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        // A normal (non-deduped) float HNSW already has real flat storage; it must not be overwritten.
        FaissHNSWIndex hnswIndex = new FaissHNSWIndex(FaissHNSWIndex.IHNF);
        FaissIndex existingFlat = mock(FaissIndex.class);
        hnswIndex.setFlatVectors(existingFlat);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();

        FaissFlatIndexFactory.maybeSetFlatBinaryIndex(idMapIndex, fieldInfo, mockReader);

        assertSame(existingFlat, hnswIndex.getFlatVectors());
    }
}
