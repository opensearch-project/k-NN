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
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;
import static org.junit.Assert.assertNotEquals;

public class VectorTransferByteTests extends TestCase {
    private static final int VECTOR_LENGTH = 20;

    @SneakyThrows
    public void testTransfer_whenCalled_thenAdded() {
        final BytesRef bytesRef1 = getByteArrayOfVectors();
        final BytesRef bytesRef2 = getByteArrayOfVectors();
        VectorTransferByte vectorTransfer = new VectorTransferByte(40);
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
        final BytesRef bytesRef = getByteArrayOfVectors();
        VectorTransferByte vectorTransfer = new VectorTransferByte(1000);

        // Verify
        assertEquals(SerializationMode.COLLECTIONS_OF_BYTES, vectorTransfer.getSerializationMode(bytesRef));
    }

    private BytesRef getByteArrayOfVectors() {
        byte[] vector = new byte[VECTOR_LENGTH];
        IntStream.range(0, VECTOR_LENGTH).forEach(index -> vector[index] = (byte) ThreadLocalRandom.current().nextInt(-128, 127));
        return new BytesRef(vector);
    }
}
