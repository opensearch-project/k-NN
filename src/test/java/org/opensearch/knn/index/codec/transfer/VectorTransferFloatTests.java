/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.jni.JNICommons;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.Assert.assertNotEquals;

public class VectorTransferFloatTests extends TestCase {
    @SneakyThrows
    public void testTransfer_whenCalled_thenAdded() {
        final BytesRef bytesRef1 = getByteArrayOfVectors(20);
        final BytesRef bytesRef2 = getByteArrayOfVectors(20);
        VectorTransferFloat vectorTransfer = new VectorTransferFloat(160);
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
        VectorTransferFloat vectorTransfer = new VectorTransferFloat(1000);

        // Verify
        assertEquals(KNNVectorSerializerFactory.getSerializerModeFromBytesRef(bytesRef), vectorTransfer.getSerializationMode(bytesRef));
    }

    private BytesRef getByteArrayOfVectors(int vectorLength) throws IOException {
        float[] vector = new float[vectorLength];
        IntStream.range(0, vectorLength).forEach(index -> vector[index] = new Random().nextFloat());

        final ByteArrayOutputStream bas = new ByteArrayOutputStream();
        final DataOutputStream ds = new DataOutputStream(bas);
        for (float f : vector) {
            ds.writeFloat(f);
        }
        final byte[] vectorAsCollectionOfFloats = bas.toByteArray();
        return new BytesRef(vectorAsCollectionOfFloats);
    }
}
