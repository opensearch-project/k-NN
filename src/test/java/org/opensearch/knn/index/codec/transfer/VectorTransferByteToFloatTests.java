/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.jni.JNICommons;

import java.io.ByteArrayInputStream;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import static org.junit.Assert.assertNotEquals;

public class VectorTransferByteToFloatTests extends TestCase {
    @SneakyThrows
    public void testTransfer_whenCalled_thenAdded() {
        final ByteArrayInputStream bais1 = getByteArrayOfVectors(20);
        final ByteArrayInputStream bais2 = getByteArrayOfVectors(20);
        VectorTransferByteToFloat vectorTransfer = new VectorTransferByteToFloat(1000);
        try {
            vectorTransfer.init(2);

            vectorTransfer.transfer(bais1);
            // flush is not called
            assertEquals(0, vectorTransfer.getVectorAddress());

            vectorTransfer.transfer(bais2);
            // flush should be called
            assertNotEquals(0, vectorTransfer.getVectorAddress());
        } finally {
            if (vectorTransfer.getVectorAddress() != 0) {
                JNICommons.freeVectorData(vectorTransfer.getVectorAddress());
            }
        }
    }

    @SneakyThrows
    public void testSerializationMode_whenCalled_thenReturn() {
        final ByteArrayInputStream bais = getByteArrayOfVectors(20);
        VectorTransferByteToFloat vectorTransfer = new VectorTransferByteToFloat(1000);

        // Verify
        assertEquals(SerializationMode.COLLECTION_OF_FLOATS, vectorTransfer.getSerializationMode(bais));
    }

    private ByteArrayInputStream getByteArrayOfVectors(int vectorLength) {
        byte[] vector = new byte[vectorLength];
        IntStream.range(0, vectorLength).forEach(index -> vector[index] = (byte) ThreadLocalRandom.current().nextInt(-128, 127));
        return new ByteArrayInputStream(vector);
    }
}
