/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.mockito.MockedStatic;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.test.OpenSearchTestCase;

import static org.mockito.Mockito.mockStatic;

public class OffHeapVectorTransferFactoryTests extends OpenSearchTestCase {

    public void testOffHeapVectorTransferFactory() {
        try (MockedStatic<KNNSettings> mockedKNNSettings = mockStatic(KNNSettings.class)) {
            mockedKNNSettings.when(KNNSettings::getVectorStreamingMemoryLimit).thenReturn(new ByteSizeValue(16));
            var floatVectorTransfer = OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 10, 10);
            assertEquals(OffHeapFloatVectorTransfer.class, floatVectorTransfer.getClass());
            assertNotSame(floatVectorTransfer, OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 10, 10));

            var halfFloatVectorTransfer = OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.HALF_FLOAT, 10, 10);
            assertEquals(OffHeapFloatVectorTransfer.class, halfFloatVectorTransfer.getClass());
            assertNotSame(halfFloatVectorTransfer, OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.HALF_FLOAT, 10, 10));

            var byteVectorTransfer = OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.BYTE, 10, 10);
            assertEquals(OffHeapByteVectorTransfer.class, byteVectorTransfer.getClass());
            assertNotSame(byteVectorTransfer, OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.BYTE, 10, 10));

            var binaryVectorTransfer = OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.BINARY, 10, 10);
            assertEquals(OffHeapBinaryVectorTransfer.class, binaryVectorTransfer.getClass());
            assertNotSame(binaryVectorTransfer, OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.BINARY, 10, 10));
        }
    }
}
