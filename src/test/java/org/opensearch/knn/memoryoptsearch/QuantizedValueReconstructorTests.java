/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizedValueReconstructor;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizedValueReconstructorFactory;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizerType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ThreadLocalRandom;

public class QuantizedValueReconstructorTests extends KNNTestCase {
    public void testReconstruct8BitDirectSigned() {
        // Index config
        final int dimension = 128;
        byte[] encodedBytes = new byte[dimension];
        float[] answerValues = new float[dimension];
        for (int i = 0; i < dimension; ++i) {
            final int value = ThreadLocalRandom.current().nextInt(-127, 128);
            answerValues[i] = value;
            // See : https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ScalarQuantizer.cpp#L844
            final byte encodedByte = (byte) (value + 128);
            encodedBytes[i] = encodedByte;
        }

        // Decoding bytes
        final FaissQuantizedValueReconstructor reconstructor = FaissQuantizedValueReconstructorFactory.create(
            FaissQuantizerType.QT_8BIT_DIRECT_SIGNED,
            dimension,
            Byte.SIZE
        );
        final byte[] buffer = new byte[dimension];
        reconstructor.reconstruct(encodedBytes, buffer);
        final float[] floatBuffer = new float[dimension];
        for (int i = 0; i < floatBuffer.length; i++) {
            floatBuffer[i] = buffer[i]; // Direct cast
        }

        // Validate results
        assertArrayEquals(answerValues, floatBuffer, 1e-6f);
    }

    public void testReconstructFP16() {
        // Index config
        final int dimension = 128;
        final int numVectorBytes = 2 * dimension;

        // Encode a vector
        final ByteBuffer buffer = ByteBuffer.allocate(dimension * Short.BYTES);
        buffer.order(ByteOrder.nativeOrder());

        // FP16 encode a vector
        final float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            final float value = ThreadLocalRandom.current().nextFloat();
            final short fp16Value = Float.floatToFloat16(value);
            buffer.putShort(fp16Value);
            vector[i] = Float.float16ToFloat(fp16Value);
        }
        final byte[] fp16EncodedBytes = buffer.array();

        // Restore encoded bytes
        final FaissQuantizedValueReconstructor reconstructor = FaissQuantizedValueReconstructorFactory.create(
            FaissQuantizerType.QT_FP16,
            dimension,
            Float.SIZE / 2
        );
        float[] restoredVector = new float[dimension];
        reconstructor.reconstruct(fp16EncodedBytes, restoredVector);

        // Validate results
        assertArrayEquals(vector, restoredVector, 1e-6f);
    }
}
