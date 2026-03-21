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

import java.lang.reflect.Field;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.SQ_CONFIG;

public class FaissFlatIndexFactoryTests extends KNNTestCase {

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenEmptyFlatVectors_thenSetsFlatIndex() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissHNSWIndex hnswIndex = new FaissHNSWIndex(FaissHNSWIndex.IHNF);
        setFlatVectors(hnswIndex, FaissEmptyIndex.INSTANCE);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");
        when(fieldInfo.getAttribute(SQ_CONFIG)).thenReturn("bits=1");

        FaissFlatIndexFactory.maybeSetFlatIndex(idMapIndex, fieldInfo, mockReader);

        assertNotNull(hnswIndex.getFlatVectors());
        assertTrue(hnswIndex.getFlatVectors() instanceof FaissScalarQuantizedFlatIndex);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenFlatVectorsNotEmptyIndex_thenDoesNotOverwrite() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissHNSWIndex hnswIndex = new FaissHNSWIndex(FaissHNSWIndex.IHNF);
        FaissIndex existing = mock(FaissIndex.class);
        setFlatVectors(hnswIndex, existing);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        FaissFlatIndexFactory.maybeSetFlatIndex(idMapIndex, fieldInfo, mockReader);

        assertSame(existing, hnswIndex.getFlatVectors());
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenNotIdMapIndex_thenNoOp() {
        FaissIndex nonIdMapIndex = mock(FaissIndex.class);
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        // Should complete without throwing
        FaissFlatIndexFactory.maybeSetFlatIndex(nonIdMapIndex, fieldInfo, mock(FlatVectorsReader.class));
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenNestedIsNotHNSW_thenNoOp() {
        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(mock(FaissIndex.class));

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        // Should complete without throwing
        FaissFlatIndexFactory.maybeSetFlatIndex(idMapIndex, fieldInfo, mock(FlatVectorsReader.class));
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenFlatIndexFactoryReturnsNull_thenThrowsIllegalState() {
        FaissHNSWIndex hnswIndex = new FaissHNSWIndex(FaissHNSWIndex.IHNF);
        setFlatVectors(hnswIndex, FaissEmptyIndex.INSTANCE);

        FaissIdMapIndex idMapIndex = new FaissIdMapIndex(FaissIdMapIndex.IXMP);
        Field nestedIndexField = FaissIdMapIndex.class.getDeclaredField("nestedIndex");
        nestedIndexField.setAccessible(true);
        nestedIndexField.set(idMapIndex, hnswIndex);

        FieldInfo nonSQFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();

        try {
            FaissFlatIndexFactory.maybeSetFlatIndex(idMapIndex, nonSQFieldInfo, mock(FlatVectorsReader.class));
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains(FaissEmptyIndex.class.getName()));
        }
    }

    @SneakyThrows
    private static void setFlatVectors(AbstractFaissHNSWIndex hnswIndex, FaissIndex flatVectors) {
        Field field = AbstractFaissHNSWIndex.class.getDeclaredField("flatVectors");
        field.setAccessible(true);
        field.set(hnswIndex, flatVectors);
    }
}
