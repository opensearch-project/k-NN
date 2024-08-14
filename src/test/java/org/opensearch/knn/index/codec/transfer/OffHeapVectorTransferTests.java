/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import lombok.SneakyThrows;
import org.apache.lucene.index.DocsWithFieldSet;
import org.junit.Before;
import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.util.Map;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING;

public class OffHeapVectorTransferTests extends KNNTestCase {

    @Mock
    ClusterSettings clusterSettings;

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();

        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);

        KNNSettings.state().setClusterService(clusterService);
    }

    @SneakyThrows
    public void testFloatTransfer() {
        // Given
        when(clusterSettings.get(KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING)).thenReturn(new ByteSizeValue(16));
        final Map<Integer, float[]> docs = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 }, 2, new float[] { 3, 4 });
        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docs.keySet().stream().sorted().forEach(docsWithFieldSet::add);

        //Transfer 1 vector
        KNNFloatVectorValues knnVectorValues = (KNNFloatVectorValues) KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, docsWithFieldSet, docs);
        knnVectorValues.nextDoc(); knnVectorValues.getVector();
        VectorTransfer vectorTransfer;

        //Transfer batch, limit == batch size
        knnVectorValues = (KNNFloatVectorValues) KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, docsWithFieldSet, docs);
        knnVectorValues.nextDoc(); knnVectorValues.getVector();
        vectorTransfer = new OffHeapFloatVectorTransfer(knnVectorValues);
        testTransferBatchVectors(vectorTransfer, new int[][] { { 0, 1 }, { 2 } }, 2);

        //Transfer batch, limit < batch size
        knnVectorValues = (KNNFloatVectorValues) KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, docsWithFieldSet, docs);
        knnVectorValues.nextDoc(); knnVectorValues.getVector();
        vectorTransfer = new OffHeapFloatVectorTransfer(knnVectorValues, 5L);
        vectorTransfer.transferBatch();
        assertNotEquals(0, vectorTransfer.getVectorAddress());
        assertArrayEquals(new int[] {0, 1, 2}, vectorTransfer.getTransferredDocsIds());

        //Transfer batch, limit > batch size
        knnVectorValues = (KNNFloatVectorValues) KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, docsWithFieldSet, docs);
        knnVectorValues.nextDoc(); knnVectorValues.getVector();
        vectorTransfer = new OffHeapFloatVectorTransfer(knnVectorValues, 1L);
        testTransferBatchVectors(vectorTransfer, new int[][] { { 0 }, { 1 }, { 2 } }, 3);
    }

    @SneakyThrows
    public void testByteTransfer() {
        // Given
        when(clusterSettings.get(KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING)).thenReturn(new ByteSizeValue(4));
        final Map<Integer, byte[]> docs = Map.of(0, new byte[] { 1, 2 }, 1, new byte[] { 2, 3 }, 2, new byte[] { 3, 4 });
        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docs.keySet().stream().sorted().forEach(docsWithFieldSet::add);

        //Transfer 1 vector
        KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, docsWithFieldSet, docs);
        knnVectorValues.nextDoc(); knnVectorValues.getVector();
        VectorTransfer vectorTransfer;

        //Transfer batch, limit == batch size
        knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, docsWithFieldSet, docs);
        knnVectorValues.nextDoc(); knnVectorValues.getVector();
        vectorTransfer = new OffHeapBytePreprocessedVectorTransfer<>(knnVectorValues);
        testTransferBatchVectors(vectorTransfer, new int[][] { { 0, 1 }, { 2 } }, 2);

        //Transfer batch, limit < batch size
        knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, docsWithFieldSet, docs);
        knnVectorValues.nextDoc(); knnVectorValues.getVector();
        vectorTransfer = new OffHeapBytePreprocessedVectorTransfer<>(knnVectorValues, 5L);
        vectorTransfer.transferBatch();
        assertNotEquals(0, vectorTransfer.getVectorAddress());
        assertArrayEquals(new int[] {0, 1, 2}, vectorTransfer.getTransferredDocsIds());

        //Transfer batch, limit > batch size
        knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, docsWithFieldSet, docs);
        knnVectorValues.nextDoc(); knnVectorValues.getVector();
        vectorTransfer = new OffHeapBytePreprocessedVectorTransfer<>(knnVectorValues, 1L);
        testTransferBatchVectors(vectorTransfer, new int[][] { { 0 }, { 1 }, { 2 } }, 3);
    }

    // TODO: Add a unit test for binary

    @SneakyThrows
    private void testTransferBatchVectors(VectorTransfer vectorTransfer, int[][] expectedDocIds, int expectedIterations) {
        long vectorAddress = 0L;
        try {
            int iteration = 0;
            while (vectorTransfer.hasNext()) {
                vectorTransfer.transferBatch();
                if (iteration != 0) {
                    assertEquals("Vector address shouldn't be different", vectorAddress, vectorTransfer.getVectorAddress());
                } else {
                    assertEquals(0, vectorAddress);
                    vectorAddress = vectorTransfer.getVectorAddress();
                }
                assertArrayEquals(expectedDocIds[iteration], vectorTransfer.getTransferredDocsIds());
                iteration++;
            }
            assertEquals(expectedIterations, iteration);
        } finally {
            vectorTransfer.close();
            assertEquals(vectorTransfer.getVectorAddress(), 0);
            assertNull(vectorTransfer.getTransferredDocsIds());
        }
    }
}
