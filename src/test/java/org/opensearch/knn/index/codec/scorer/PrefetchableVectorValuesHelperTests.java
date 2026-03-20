/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.store.IndexInput;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexScalarQuantizedFlat;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissFloatVectorValues;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class PrefetchableVectorValuesHelperTests extends KNNTestCase {

    private final int[] nodes = { 0, 1, 2 };
    private final int numNodes = 3;

    public void testMayBeDoPrefetch_whenFloatVectorValuesImpl_thenPrefetchesViaHasIndexSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 512;

        FaissFloatVectorValues floatImpl = mock(FaissFloatVectorValues.class);
        when(floatImpl.getSlice()).thenReturn(mockSlice);
        when(floatImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.mayBeDoPrefetch(floatImpl, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testMayBeDoPrefetch_whenQuantizedFloatVectorValuesImpl_thenPrefetchesViaHasIndexSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 512;

        FaissIndexScalarQuantizedFlat.FloatVectorValuesImpl floatImpl = mock(FaissIndexScalarQuantizedFlat.FloatVectorValuesImpl.class);
        when(floatImpl.getSlice()).thenReturn(mockSlice);
        when(floatImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.mayBeDoPrefetch(floatImpl, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testMayBeDoPrefetch_whenByteVectorValuesImpl_thenPrefetchesViaHasIndexSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 64;

        FaissIndexBinaryFlat.ByteVectorValuesImpl binaryImpl = mock(FaissIndexBinaryFlat.ByteVectorValuesImpl.class);
        when(binaryImpl.getSlice()).thenReturn(mockSlice);
        when(binaryImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.mayBeDoPrefetch(binaryImpl, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testMayBeDoPrefetch_whenOffHeapFloatVectorValues_thenCallsPrefetchHelper() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 16;

        OffHeapFloatVectorValues hasSliceValues = mock(OffHeapFloatVectorValues.class);
        when(hasSliceValues.getSlice()).thenReturn(mockSlice);
        when(hasSliceValues.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.mayBeDoPrefetch(hasSliceValues, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testMayBeDoPrefetch_whenUnsupportedType_thenNoException() throws IOException {
        KnnVectorValues unsupported = mock(FloatVectorValues.class);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.mayBeDoPrefetch(unsupported, nodes, numNodes);

            mockedPrefetchHelper.verifyNoInteractions();
        }
    }

    public void testMayBeDoPrefetch_whenZeroNumNodes_thenDelegatesAsIs() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 64;

        FaissIndexBinaryFlat.ByteVectorValuesImpl binaryImpl = mock(FaissIndexBinaryFlat.ByteVectorValuesImpl.class);
        when(binaryImpl.getSlice()).thenReturn(mockSlice);
        when(binaryImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.mayBeDoPrefetch(binaryImpl, nodes, 0);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, 0));
        }
    }
}
