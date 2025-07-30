/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.jni.JNICommons;

/**
 * Class implements KNNVectorSerializer based on serialization/deserialization of float array
 * as a collection of individual half-precision values
 */
public class KNNVectorAsCollectionOfHalfFloatsSerializer implements KNNVectorSerializer {
    private static final int BYTES_IN_HALF_FLOAT = 2;

    public static final KNNVectorAsCollectionOfHalfFloatsSerializer INSTANCE = new KNNVectorAsCollectionOfHalfFloatsSerializer();

    /**
     * Converts float[] to byte[] using JNI (FP32 to FP16).
     * @param input the float[] to be serialized into half-precision format
     * @return a byte[] containing the float16-encoded data
     */
    @Override
    public byte[] floatToByteArray(float[] input) {
        if (input == null) {
            throw new IllegalArgumentException("Input float array is null. Cannot convert to FP16.");
        }
        byte[] output = new byte[input.length * BYTES_IN_HALF_FLOAT];
        floatToByteArray(input, output, input.length);
        return output;
    }

    /**
     * Converts float[] to byte[] using JNI (FP32 to FP16).
     * @param input the float[] to be serialized into half-precision format
     * @param output byte[] containing the float16-encoded data
     * @param count number of floats to serialize
     */
    public void floatToByteArray(float[] input, byte[] output, int count) {
        if (input == null || output == null) {
            throw new IllegalArgumentException("Input/output buffers cannot be null.");
        }
        if (output.length != input.length * BYTES_IN_HALF_FLOAT) {
            throw new IllegalArgumentException("Output buffer size mismatch. Must be 2x input length.");
        }
        JNICommons.convertFP32ToFP16(input, output, count);
    }

    /**
     * Converts a BytesRef-wrapped byte array (encoded as float16) back into a float array.
     *
     * @param bytesRef the BytesRef containing float16-encoded vector data
     * @return a float containing the decoded float32 values
     */
    @Override
    public float[] byteToFloatArray(BytesRef bytesRef) {
        if (bytesRef == null || bytesRef.length % BYTES_IN_HALF_FLOAT != 0) {
            throw new IllegalArgumentException("Byte stream cannot be deserialized to array of half-floats");
        }

        int count = bytesRef.length / BYTES_IN_HALF_FLOAT;
        float[] output = new float[count];
        byteToFloatArray(bytesRef.bytes, output, count, 0);
        return output;
    }

    /**
     * Converts byte[] to float[] using JNI (FP16 to FP32).
     * @param input the byte[] containing half-precision encoded data
     * @param output a float[] containing the decoded float32 values
     * @param count number of floats to deserialize
     * @param offset offset in the output array where deserialization should start
     */
    public void byteToFloatArray(byte[] input, float[] output, int count, int offset) {
        if (input == null || output == null) {
            throw new IllegalArgumentException("Input/output buffers cannot be null.");
        }
        if (input.length % BYTES_IN_HALF_FLOAT != 0) {
            throw new IllegalArgumentException(
                    "Invalid byte array length ("
                            + input.length
                            + "). Must be a multiple of "
                            + BYTES_IN_HALF_FLOAT
                            + " to represent float16 values."
            );
        }
        JNICommons.convertFP16ToFP32(input, output, count, offset);
    }
}
