/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.jni.JNICommons;

import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertNotEquals;

public class VectorTransferByteTests extends TestCase {
    @SneakyThrows
    public void testTransfer_whenCalled_thenAdded() {
        final BytesRef bytesRef1 = getByteArrayOfVectors(20);
        final BytesRef bytesRef2 = getByteArrayOfVectors(20);
        VectorTransferByte vectorTransfer = new VectorTransferByte(1000);
        try {
            vectorTransfer.init(2);

            vectorTransfer.transfer(bytesRef1);
            // flush is not called
            assertEquals(0, vectorTransfer.getVectorAddress());

            vectorTransfer.transfer(bytesRef2);
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
        final BytesRef bytesRef = getByteArrayOfVectors(20);
        VectorTransferByte vectorTransfer = new VectorTransferByte(1000);

        // Verify
        assertEquals(SerializationMode.COLLECTIONS_OF_BYTES, vectorTransfer.getSerializationMode(bytesRef));
    }

    private BytesRef getByteArrayOfVectors(int vectorLength) throws IOException {
        byte[] vector = new byte[vectorLength];
        new Random().nextBytes(vector);
        return new BytesRef(vector);
    }
}
