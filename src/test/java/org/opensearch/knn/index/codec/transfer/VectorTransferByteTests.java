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
import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertNotEquals;

public class VectorTransferByteTests extends TestCase {
    @SneakyThrows
    public void testTransfer_whenCalled_thenAdded() {
        final ByteArrayInputStream bais1 = getByteArrayOfVectors(20);
        final ByteArrayInputStream bais2 = getByteArrayOfVectors(20);
        VectorTransferByte vectorTransfer = new VectorTransferByte(1000);
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
        VectorTransferByte vectorTransfer = new VectorTransferByte(1000);

        // Verify
        assertEquals(SerializationMode.COLLECTIONS_OF_BYTES, vectorTransfer.getSerializationMode(bais));
    }

    private ByteArrayInputStream getByteArrayOfVectors(int vectorLength) throws IOException {
        byte[] vector = new byte[vectorLength];
        new Random().nextBytes(vector);
        return new ByteArrayInputStream(vector);
    }
}
