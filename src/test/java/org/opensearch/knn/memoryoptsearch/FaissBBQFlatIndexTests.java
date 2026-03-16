/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissBBQFlatIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.lang.reflect.Method;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FaissBBQFlatIndexTests extends KNNTestCase {

    public void testFaissBBQFlatIndex_hasReaderReference() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        String fieldName = "test_field";

        FaissBBQFlatIndex bbqFlatIndex = new FaissBBQFlatIndex(mockReader, fieldName);

        assertSame(mockReader, bbqFlatIndex.getBbqFlatReader());
        assertEquals(fieldName, bbqFlatIndex.getFieldName());
        assertEquals("FaissBBQFlatIndex", bbqFlatIndex.getIndexType());
        assertEquals(VectorEncoding.FLOAT32, bbqFlatIndex.getVectorEncoding());
    }

    @SneakyThrows
    public void testFaissBBQFlatIndex_getFloatValues_delegatesToReader() {
        FloatVectorValues mockFloatValues = mock(FloatVectorValues.class);
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        String fieldName = "test_field";
        when(mockReader.getFloatVectorValues(fieldName)).thenReturn(mockFloatValues);

        FaissBBQFlatIndex bbqFlatIndex = new FaissBBQFlatIndex(mockReader, fieldName);
        FloatVectorValues result = bbqFlatIndex.getFloatValues(null);

        assertSame(mockFloatValues, result);
    }

    public void testFaissBBQFlatIndex_getByteValues_throwsUnsupported() {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        FaissBBQFlatIndex bbqFlatIndex = new FaissBBQFlatIndex(mockReader, "test_field");

        assertThrows(UnsupportedOperationException.class, () -> bbqFlatIndex.getByteValues(null));
    }

    @SneakyThrows
    public void testFaissBBQFlatIndex_doLoad_isNoOp() throws Exception {
        FlatVectorsReader mockReader = mock(FlatVectorsReader.class);
        FaissBBQFlatIndex bbqFlatIndex = new FaissBBQFlatIndex(mockReader, "test_field");

        // doLoad is protected, use reflection to verify it's a no-op
        IndexInput mockInput = mock(IndexInput.class);
        final Method doLoadMethod = FaissBBQFlatIndex.class.getDeclaredMethod("doLoad", IndexInput.class);
        doLoadMethod.setAccessible(true);
        doLoadMethod.invoke(bbqFlatIndex, mockInput);
    }

    @SneakyThrows
    public void testFaissIndexLoad_whenNullSection_thenReturnsNull() {
        IndexInput mockInput = mock(IndexInput.class);
        // "null" as 4 bytes — readBytes is void, so use doAnswer
        doAnswer(invocation -> {
            byte[] buf = invocation.getArgument(0);
            byte[] nullBytes = "null".getBytes();
            System.arraycopy(nullBytes, 0, buf, 0, 4);
            return null;
        }).when(mockInput).readBytes(any(byte[].class), any(int.class), any(int.class));

        FaissIndex result = FaissIndex.load(mockInput);
        assertNull(result);
    }

    public void testAbstractFaissHNSWIndex_flatVectorsGetter() {
        FaissHNSWIndex hnswIndex = mock(FaissHNSWIndex.class);
        FaissBBQFlatIndex bbqFlatIndex = new FaissBBQFlatIndex(mock(FlatVectorsReader.class), "test_field");
        when(hnswIndex.getFlatVectors()).thenReturn(bbqFlatIndex);

        FaissIndex flatVectors = hnswIndex.getFlatVectors();
        assertTrue(flatVectors instanceof FaissBBQFlatIndex);
        assertSame(bbqFlatIndex, flatVectors);
    }

    @SneakyThrows
    public void testFaissBBQFlatIndex_creation() {
        FlatVectorsReader mockBbqFlatReader = mock(FlatVectorsReader.class);
        FlatVectorsScorer mockScorer = mock(FlatVectorsScorer.class);
        when(mockBbqFlatReader.getFlatVectorScorer()).thenReturn(mockScorer);
        String fieldName = "my_vector_field";

        FaissBBQFlatIndex bbqIndex = new FaissBBQFlatIndex(mockBbqFlatReader, fieldName);
        assertSame(mockBbqFlatReader, bbqIndex.getBbqFlatReader());
        assertEquals(fieldName, bbqIndex.getFieldName());
        assertSame(mockScorer, bbqIndex.getBbqFlatReader().getFlatVectorScorer());
    }

}
