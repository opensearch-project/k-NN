/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;

import java.util.List;

public class OffHeapVectorTransferTests extends KNNTestCase {

    @SneakyThrows
    public void testFloatTransfer() {
        List<float[]> vectors = List.of(
            new float[] { 0.1f, 0.2f },
            new float[] { 0.2f, 0.3f },
            new float[] { 0.3f, 0.4f },
            new float[] { 0.3f, 0.4f },
            new float[] { 0.3f, 0.4f }
        );

        OffHeapFloatVectorTransfer vectorTransfer = new OffHeapFloatVectorTransfer(2);
        long vectorAddress = 0;
        assertFalse(vectorTransfer.transfer(vectors.get(0), false));
        assertEquals(0, vectorTransfer.getVectorAddress());
        assertTrue(vectorTransfer.transfer(vectors.get(1), false));
        vectorAddress = vectorTransfer.getVectorAddress();
        assertFalse(vectorTransfer.transfer(vectors.get(2), false));
        assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
        assertTrue(vectorTransfer.transfer(vectors.get(3), false));
        assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
        assertFalse(vectorTransfer.transfer(vectors.get(4), false));
        assertTrue(vectorTransfer.flush(false));
        vectorTransfer.close();
    }

    @SneakyThrows
    public void testByteTransfer() {
        List<byte[]> vectors = List.of(
            new byte[] { 0, 1 },
            new byte[] { 2, 3 },
            new byte[] { 4, 5 },
            new byte[] { 6, 7 },
            new byte[] { 8, 9 }
        );

        OffHeapByteVectorTransfer vectorTransfer = new OffHeapByteVectorTransfer(2);
        long vectorAddress = 0;
        assertFalse(vectorTransfer.transfer(vectors.get(0), false));
        assertEquals(0, vectorTransfer.getVectorAddress());
        assertTrue(vectorTransfer.transfer(vectors.get(1), false));
        vectorAddress = vectorTransfer.getVectorAddress();
        assertFalse(vectorTransfer.transfer(vectors.get(2), false));
        assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
        assertTrue(vectorTransfer.transfer(vectors.get(3), false));
        assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
        assertFalse(vectorTransfer.transfer(vectors.get(4), false));
        assertTrue(vectorTransfer.flush(false));
        vectorTransfer.close();
    }

    @SneakyThrows
    public void testBinaryTransfer() {
        List<byte[]> vectors = List.of(
            new byte[] { 0, 1 },
            new byte[] { 2, 3 },
            new byte[] { 4, 5 },
            new byte[] { 6, 7 },
            new byte[] { 8, 9 }
        );

        OffHeapBinaryVectorTransfer vectorTransfer = new OffHeapBinaryVectorTransfer(2);
        long vectorAddress = 0;
        assertFalse(vectorTransfer.transfer(vectors.get(0), false));
        assertEquals(0, vectorTransfer.getVectorAddress());
        assertTrue(vectorTransfer.transfer(vectors.get(1), false));
        vectorAddress = vectorTransfer.getVectorAddress();
        assertFalse(vectorTransfer.transfer(vectors.get(2), false));
        assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
        assertTrue(vectorTransfer.transfer(vectors.get(3), false));
        assertEquals(vectorAddress, vectorTransfer.getVectorAddress());
        assertFalse(vectorTransfer.transfer(vectors.get(4), false));
        assertTrue(vectorTransfer.flush(false));
        vectorTransfer.close();
    }
}
