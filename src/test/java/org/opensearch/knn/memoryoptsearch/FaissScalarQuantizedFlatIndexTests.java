/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissScalarQuantizedFlatIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.lang.reflect.Method;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FaissScalarQuantizedFlatIndexTests extends KNNTestCase {

    public void testFaissScalarQuantizedFlatIndex_hasReaderReference() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        String fieldName = "test_field";

        FaissScalarQuantizedFlatIndex sqFlatIndex = new FaissScalarQuantizedFlatIndex(mockReader, fieldName);

        assertSame(mockReader, sqFlatIndex.getFlatVectorsReader());
        assertEquals(fieldName, sqFlatIndex.getFieldName());
        assertEquals("FaissScalarQuantizedFlatIndex", sqFlatIndex.getIndexType());
        assertEquals(VectorEncoding.FLOAT32, sqFlatIndex.getVectorEncoding());
    }

    @SneakyThrows
    public void testFaissScalarQuantizedFlatIndex_getFloatValues_delegatesToReader() {
        FloatVectorValues mockFloatValues = mock(FloatVectorValues.class);
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        String fieldName = "test_field";
        when(mockReader.getFloatVectorValues(fieldName)).thenReturn(mockFloatValues);

        FaissScalarQuantizedFlatIndex sqFlatIndex = new FaissScalarQuantizedFlatIndex(mockReader, fieldName);
        FloatVectorValues result = sqFlatIndex.getFloatValues(null);

        assertSame(mockFloatValues, result);
    }

    public void testFaissScalarQuantizedFlatIndex_getByteValues_throwsUnsupported() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        FaissScalarQuantizedFlatIndex sqFlatIndex = new FaissScalarQuantizedFlatIndex(mockReader, "test_field");

        assertThrows(UnsupportedOperationException.class, () -> sqFlatIndex.getByteValues(null));
    }

    @SneakyThrows
    public void testFaissScalarQuantizedFlatIndex_doLoad_isNoOp() throws Exception {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        FaissScalarQuantizedFlatIndex sqFlatIndex = new FaissScalarQuantizedFlatIndex(mockReader, "test_field");

        IndexInput mockInput = mock(IndexInput.class);
        final Method doLoadMethod = FaissScalarQuantizedFlatIndex.class.getDeclaredMethod("doLoad", IndexInput.class);
        doLoadMethod.setAccessible(true);
        doLoadMethod.invoke(sqFlatIndex, mockInput);
    }

    public void testAbstractFaissHNSWIndex_flatVectorsGetter() {
        FaissHNSWIndex hnswIndex = mock(FaissHNSWIndex.class);
        FaissScalarQuantizedFlatIndex sqFlatIndex = new FaissScalarQuantizedFlatIndex(mock(FlatVectorsReader.class), "test_field");
        when(hnswIndex.getFlatVectors()).thenReturn(sqFlatIndex);

        FaissIndex flatVectors = hnswIndex.getFlatVectors();
        assertTrue(flatVectors instanceof FaissScalarQuantizedFlatIndex);
        assertSame(sqFlatIndex, flatVectors);
    }
}
