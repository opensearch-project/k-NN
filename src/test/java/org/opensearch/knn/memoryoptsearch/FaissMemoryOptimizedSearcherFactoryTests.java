/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.AbstractFaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissBBQFlatIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcherFactory;

import java.lang.reflect.Method;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class FaissMemoryOptimizedSearcherFactoryTests extends KNNTestCase {

    @SneakyThrows
    public void testMaybeSetFlatVectorsFromReader_whenNullFlatVectors_thenSetsBBQFlatIndex() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        when(mockReader.getFlatVectorScorer()).thenReturn(mock(FlatVectorsScorer.class));

        AbstractFaissHNSWIndex hnswIndex = mock(FaissHNSWIndex.class);
        when(hnswIndex.getFlatVectors()).thenReturn(null);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        invokeMaybeSetFlatVectorsFromReader(idMapIndex, mockReader, fieldInfo);

        verify(hnswIndex).setFlatVectors(any(FaissBBQFlatIndex.class));
    }

    @SneakyThrows
    public void testMaybeSetFlatVectorsFromReader_whenFlatVectorsAlreadySet_thenDoesNotOverwrite() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);

        AbstractFaissHNSWIndex hnswIndex = mock(FaissHNSWIndex.class);
        when(hnswIndex.getFlatVectors()).thenReturn(mock(FaissIndex.class));

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(hnswIndex);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        invokeMaybeSetFlatVectorsFromReader(idMapIndex, mockReader, fieldInfo);

        verify(hnswIndex, never()).setFlatVectors(any());
    }

    @SneakyThrows
    public void testMaybeSetFlatVectorsFromReader_whenNotIdMapIndex_thenNoOp() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        FaissIndex nonIdMapIndex = mock(FaissIndex.class);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        // Should not throw — just a no-op
        invokeMaybeSetFlatVectorsFromReader(nonIdMapIndex, mockReader, fieldInfo);
    }

    @SneakyThrows
    public void testMaybeSetFlatVectorsFromReader_whenNestedIsNotHNSW_thenNoOp() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        FaissIndex nonHnswNested = mock(FaissIndex.class);

        FaissIdMapIndex idMapIndex = mock(FaissIdMapIndex.class);
        when(idMapIndex.getNestedIndex()).thenReturn(nonHnswNested);

        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getName()).thenReturn("test_field");

        // Should not throw — nested is not HNSW so nothing to wire
        invokeMaybeSetFlatVectorsFromReader(idMapIndex, mockReader, fieldInfo);
    }

    @SneakyThrows
    private static void invokeMaybeSetFlatVectorsFromReader(FaissIndex faissIndex, FlatVectorsReader reader, FieldInfo fieldInfo) {
        FaissMemoryOptimizedSearcherFactory factory = new FaissMemoryOptimizedSearcherFactory();
        Method method = FaissMemoryOptimizedSearcherFactory.class.getDeclaredMethod(
            "maybeSetFlatVectorsFromReader",
            FaissIndex.class,
            FlatVectorsReader.class,
            FieldInfo.class
        );
        method.setAccessible(true);
        method.invoke(factory, faissIndex, reader, fieldInfo);
    }
}
