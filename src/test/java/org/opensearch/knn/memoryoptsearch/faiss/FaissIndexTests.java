/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.FAISS_BBQ_CONFIG;

public class FaissIndexTests extends KNNTestCase {

    @SneakyThrows
    public void testLoad_whenNullSection_thenReturnsFaissEmptyIndex() {
        IndexInput mockInput = mock(IndexInput.class);
        doAnswer(invocation -> {
            byte[] buf = invocation.getArgument(0);
            byte[] nullBytes = "null".getBytes();
            System.arraycopy(nullBytes, 0, buf, 0, 4);
            return null;
        }).when(mockInput).readBytes(any(byte[].class), any(int.class), any(int.class));

        FaissIndex result = FaissIndex.load(mockInput);
        assertSame(FaissEmptyIndex.INSTANCE, result);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenEmptyFlatVectors_thenSetsSQFlatIndex() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        FaissHNSWIndex hnswIndex = new FaissHNSWIndex(FaissHNSWIndex.IHNF);
        setFlatVectors(hnswIndex, FaissEmptyIndex.INSTANCE);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");
        when(fieldInfo.getAttribute(FAISS_BBQ_CONFIG)).thenReturn("sq_config_value");

        invokeMaybeSetFlatIndex(idMapIndex, fieldInfo, mockReader);

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

        invokeMaybeSetFlatIndex(idMapIndex, fieldInfo, mockReader);

        assertSame(existing, hnswIndex.getFlatVectors());
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenNotIdMapIndex_thenNoOp() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        FaissIndex nonIdMapIndex = mock(FaissIndex.class);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        invokeMaybeSetFlatIndex(nonIdMapIndex, fieldInfo, mockReader);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenNestedIsNotHNSW_thenNoOp() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        FaissIndex nonHnswNested = mock(FaissIndex.class);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(nonHnswNested);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        invokeMaybeSetFlatIndex(idMapIndex, fieldInfo, mockReader);
    }

    @SneakyThrows
    public void testMaybeSetFlatIndex_whenFlatIndexFactoryReturnsNull_thenThrowsIllegalState() {
        FaissHNSWIndex hnswIndex = new FaissHNSWIndex(FaissHNSWIndex.IHNF);
        Field flatVectorsField = AbstractFaissHNSWIndex.class.getDeclaredField("flatVectors");
        flatVectorsField.setAccessible(true);
        flatVectorsField.set(hnswIndex, FaissEmptyIndex.INSTANCE);

        FaissIdMapIndex idMapIndex = new FaissIdMapIndex(FaissIdMapIndex.IXMP);
        Field nestedIndexField = FaissIdMapIndex.class.getDeclaredField("nestedIndex");
        nestedIndexField.setAccessible(true);
        nestedIndexField.set(idMapIndex, hnswIndex);

        FieldInfo nonBbqFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();

        Method maybeSetFlatIndex = FaissIndex.class.getDeclaredMethod(
            "maybeSetFlatIndex",
            FaissIndex.class,
            FieldInfo.class,
            FlatVectorsReader.class
        );
        maybeSetFlatIndex.setAccessible(true);

        try {
            maybeSetFlatIndex.invoke(null, idMapIndex, nonBbqFieldInfo, mock(FlatVectorsReader.class));
            fail("Expected IllegalStateException");
        } catch (java.lang.reflect.InvocationTargetException e) {
            assertTrue(e.getCause() instanceof IllegalStateException);
            assertTrue(e.getCause().getMessage().contains(FaissEmptyIndex.class.getName()));
        }
    }

    @SneakyThrows
    private static void setFlatVectors(AbstractFaissHNSWIndex hnswIndex, FaissIndex flatVectors) {
        Field field = AbstractFaissHNSWIndex.class.getDeclaredField("flatVectors");
        field.setAccessible(true);
        field.set(hnswIndex, flatVectors);
    }

    @SneakyThrows
    private static void invokeMaybeSetFlatIndex(FaissIndex faissIndex, FieldInfo fieldInfo, FlatVectorsReader reader) {
        Method method = FaissIndex.class.getDeclaredMethod("maybeSetFlatIndex", FaissIndex.class, FieldInfo.class, FlatVectorsReader.class);
        method.setAccessible(true);
        method.invoke(null, faissIndex, fieldInfo, reader);
    }
}
