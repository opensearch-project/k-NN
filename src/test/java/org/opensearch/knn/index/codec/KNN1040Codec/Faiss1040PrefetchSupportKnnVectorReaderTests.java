/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class Faiss1040PrefetchSupportKnnVectorReaderTests extends KNNTestCase {

    @SneakyThrows
    public void testConstructor_thenUsesScorerFromDelegate() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FlatVectorsScorer expectedScorer = mock(FlatVectorsScorer.class);
        when(delegate.getFlatVectorScorer()).thenReturn(expectedScorer);

        Faiss1040PrefetchSupportKnnVectorReader reader = new Faiss1040PrefetchSupportKnnVectorReader(delegate);
        assertSame(expectedScorer, reader.getFlatVectorScorer());
    }

    @SneakyThrows
    public void testGetRandomVectorScorerFloat_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);
        float[] target = { 1.0f, 2.0f };
        when(delegate.getRandomVectorScorer("field", target)).thenReturn(expectedScorer);

        Faiss1040PrefetchSupportKnnVectorReader reader = new Faiss1040PrefetchSupportKnnVectorReader(delegate);
        assertSame(expectedScorer, reader.getRandomVectorScorer("field", target));
        verify(delegate).getRandomVectorScorer("field", target);
    }

    @SneakyThrows
    public void testGetRandomVectorScorerByte_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);
        byte[] target = { 1, 2 };
        when(delegate.getRandomVectorScorer("field", target)).thenReturn(expectedScorer);

        Faiss1040PrefetchSupportKnnVectorReader reader = new Faiss1040PrefetchSupportKnnVectorReader(delegate);
        assertSame(expectedScorer, reader.getRandomVectorScorer("field", target));
        verify(delegate).getRandomVectorScorer("field", target);
    }

    @SneakyThrows
    public void testCheckIntegrity_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        Faiss1040PrefetchSupportKnnVectorReader reader = new Faiss1040PrefetchSupportKnnVectorReader(delegate);
        reader.checkIntegrity();
        verify(delegate).checkIntegrity();
    }

    @SneakyThrows
    public void testGetFloatVectorValues_thenReturnsWrappedValuesWithHasIndexSlice() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FloatVectorValues mockFvv = mock(FloatVectorValues.class);
        QuantizedByteVectorValues mockQbvv = mock(QuantizedByteVectorValues.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(mockFvv);

        try (MockedStatic<Faiss1040ScalarQuantizedUtils> mockedUtils = mockStatic(Faiss1040ScalarQuantizedUtils.class)) {
            mockedUtils.when(() -> Faiss1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(any())).thenReturn(mockQbvv);

            Faiss1040PrefetchSupportKnnVectorReader reader = new Faiss1040PrefetchSupportKnnVectorReader(delegate);
            FloatVectorValues result = reader.getFloatVectorValues("field");

            assertTrue("Result should implement HasIndexSlice", result instanceof HasIndexSlice);
            assertTrue(
                "Result should be ScalarQuantizedFloatVectorValuesWithIndexInputSlice",
                result instanceof ScalarQuantizedFloatVectorValuesWithIndexInputSlice
            );
            verify(delegate).getFloatVectorValues("field");
            mockedUtils.verify(() -> Faiss1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(mockFvv));
        }
    }

    @SneakyThrows
    public void testGetByteVectorValues_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        ByteVectorValues expectedValues = mock(ByteVectorValues.class);
        when(delegate.getByteVectorValues("field")).thenReturn(expectedValues);

        Faiss1040PrefetchSupportKnnVectorReader reader = new Faiss1040PrefetchSupportKnnVectorReader(delegate);
        assertSame(expectedValues, reader.getByteVectorValues("field"));
        verify(delegate).getByteVectorValues("field");
    }

    @SneakyThrows
    public void testClose_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        Faiss1040PrefetchSupportKnnVectorReader reader = new Faiss1040PrefetchSupportKnnVectorReader(delegate);
        reader.close();
        verify(delegate).close();
    }

    @SneakyThrows
    public void testRamBytesUsed_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        when(delegate.ramBytesUsed()).thenReturn(12345L);

        Faiss1040PrefetchSupportKnnVectorReader reader = new Faiss1040PrefetchSupportKnnVectorReader(delegate);
        assertEquals(12345L, reader.ramBytesUsed());
        verify(delegate).ramBytesUsed();
    }
}
