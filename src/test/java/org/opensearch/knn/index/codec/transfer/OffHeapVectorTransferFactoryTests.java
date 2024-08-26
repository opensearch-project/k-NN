/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.test.OpenSearchTestCase;

public class OffHeapVectorTransferFactoryTests extends OpenSearchTestCase {

    public void testOffHeapVectorTransferFactory() {
        var floatVectorTransfer = OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 10);
        assertEquals(OffHeapFloatVectorTransfer.class, floatVectorTransfer.getClass());
        assertNotSame(floatVectorTransfer, OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.FLOAT, 10));

        var byteVectorTransfer = OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.BYTE, 10);
        assertEquals(OffHeapByteVectorTransfer.class, byteVectorTransfer.getClass());
        assertNotSame(byteVectorTransfer, OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.BYTE, 10));

        var binaryVectorTransfer = OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.BINARY, 10);
        assertEquals(OffHeapBinaryVectorTransfer.class, binaryVectorTransfer.getClass());
        assertNotSame(binaryVectorTransfer, OffHeapVectorTransferFactory.getVectorTransfer(VectorDataType.BINARY, 10));
    }
}
