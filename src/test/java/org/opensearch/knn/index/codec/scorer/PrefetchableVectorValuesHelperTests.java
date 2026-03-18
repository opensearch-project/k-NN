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
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexFloatFlat;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class PrefetchableVectorValuesHelperTests extends KNNTestCase {

    private final int[] nodes = { 0, 1, 2 };
    private final int numNodes = 3;

    public void testMayBeDoPrefetch_whenFloatVectorValuesImpl_thenDelegates() throws IOException {
        FaissIndexFloatFlat.FloatVectorValuesImpl floatImpl = mock(FaissIndexFloatFlat.FloatVectorValuesImpl.class);

        PrefetchableVectorValuesHelper.mayBeDoPrefetch(floatImpl, nodes, numNodes);

        verify(floatImpl).prefetch(nodes, numNodes);
    }

    public void testMayBeDoPrefetch_whenByteVectorValuesImpl_thenDelegates() throws IOException {
        FaissIndexBinaryFlat.ByteVectorValuesImpl binaryImpl = mock(FaissIndexBinaryFlat.ByteVectorValuesImpl.class);

        PrefetchableVectorValuesHelper.mayBeDoPrefetch(binaryImpl, nodes, numNodes);

        verify(binaryImpl).prefetch(nodes, numNodes);
    }

    public void testMayBeDoPrefetch_whenHasIndexSlice_thenCallsPrefetchHelper() throws IOException {
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

        // Should not throw, just logs
        PrefetchableVectorValuesHelper.mayBeDoPrefetch(unsupported, nodes, numNodes);
    }

    public void testMayBeDoPrefetch_whenNullNodes_thenDelegatesAsIs() throws IOException {
        FaissIndexFloatFlat.FloatVectorValuesImpl floatImpl = mock(FaissIndexFloatFlat.FloatVectorValuesImpl.class);

        PrefetchableVectorValuesHelper.mayBeDoPrefetch(floatImpl, null, 0);

        verify(floatImpl).prefetch(null, 0);
    }

    public void testMayBeDoPrefetch_whenZeroNumNodes_thenDelegatesAsIs() throws IOException {
        FaissIndexBinaryFlat.ByteVectorValuesImpl binaryImpl = mock(FaissIndexBinaryFlat.ByteVectorValuesImpl.class);

        PrefetchableVectorValuesHelper.mayBeDoPrefetch(binaryImpl, nodes, 0);

        verify(binaryImpl).prefetch(nodes, 0);
    }
}
