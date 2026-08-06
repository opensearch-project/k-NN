/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class Faiss1040ScalarQuantizedFlatVectorsReaderTests extends KNNTestCase {

    @SneakyThrows
    public void testConstructor_thenUsesScorerFromDelegate() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FlatVectorsScorer expectedScorer = mock(FlatVectorsScorer.class);
        when(delegate.getFlatVectorScorer("test_field")).thenReturn(expectedScorer);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertSame(expectedScorer, reader.getFlatVectorScorer("test_field"));
    }

    @SneakyThrows
    public void testGetRandomVectorScorerFloat_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);
        float[] target = { 1.0f, 2.0f };
        when(delegate.getRandomVectorScorer("field", target)).thenReturn(expectedScorer);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertSame(expectedScorer, reader.getRandomVectorScorer("field", target));
        verify(delegate).getRandomVectorScorer("field", target);
    }

    @SneakyThrows
    public void testGetRandomVectorScorerByte_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);
        byte[] target = { 1, 2 };
        when(delegate.getRandomVectorScorer("field", target)).thenReturn(expectedScorer);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertSame(expectedScorer, reader.getRandomVectorScorer("field", target));
        verify(delegate).getRandomVectorScorer("field", target);
    }

    @SneakyThrows
    public void testCheckIntegrity_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        reader.checkIntegrity();
        verify(delegate).checkIntegrity();
    }

    @SneakyThrows
    public void testGetFloatVectorValues_thenReturnsWrappedValuesWithHasIndexSlice() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FloatVectorValues mockFvv = mock(FloatVectorValues.class);
        QuantizedByteVectorValues mockQbvv = mock(QuantizedByteVectorValues.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(mockFvv);
        when(mockFvv.size()).thenReturn(1);

        try (MockedStatic<KNN1040ScalarQuantizedUtils> mockedUtils = mockStatic(KNN1040ScalarQuantizedUtils.class)) {
            mockedUtils.when(() -> KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(any())).thenReturn(mockQbvv);

            Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
            FloatVectorValues result = reader.getFloatVectorValues("field");

            assertTrue("Result should implement HasIndexSlice", result instanceof HasIndexSlice);
            assertTrue("Result should be ScalarQuantizedFloatVectorValues", result instanceof ScalarQuantizedFloatVectorValues);
            verify(delegate).getFloatVectorValues("field");
            mockedUtils.verify(() -> KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(mockFvv));
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_whenEmpty_thenReturnsWrappedEmptyValuesWithoutExtraction() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FloatVectorValues emptyValues = mock(FloatVectorValues.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(emptyValues);
        when(emptyValues.size()).thenReturn(0);

        try (MockedStatic<KNN1040ScalarQuantizedUtils> mockedUtils = mockStatic(KNN1040ScalarQuantizedUtils.class)) {
            Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
            FloatVectorValues result = reader.getFloatVectorValues("field");

            assertTrue(result instanceof ScalarQuantizedFloatVectorValues);
            assertTrue(result instanceof HasIndexSlice);
            assertNull(((HasIndexSlice) result).getSlice());
            assertEquals(0, result.size());
            verify(delegate).getFloatVectorValues("field");
            mockedUtils.verifyNoInteractions();
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_whenDelegateReturnsNull_thenReturnsNull() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(null);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);

        assertNull(reader.getFloatVectorValues("field"));
        verify(delegate).getFloatVectorValues("field");
    }

    @SneakyThrows
    public void testGetByteVectorValues_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        ByteVectorValues expectedValues = mock(ByteVectorValues.class);
        when(delegate.getByteVectorValues("field")).thenReturn(expectedValues);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertSame(expectedValues, reader.getByteVectorValues("field"));
        verify(delegate).getByteVectorValues("field");
    }

    @SneakyThrows
    public void testClose_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        reader.close();
        verify(delegate).close();
    }

    @SneakyThrows
    public void testRamBytesUsed_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        when(delegate.ramBytesUsed()).thenReturn(12345L);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertEquals(12345L, reader.ramBytesUsed());
        verify(delegate).ramBytesUsed();
    }
}
