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

import java.lang.reflect.Field;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.SQ_CONFIG;

public class FaissFlatIndexFactoryTests extends KNNTestCase {

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
}
