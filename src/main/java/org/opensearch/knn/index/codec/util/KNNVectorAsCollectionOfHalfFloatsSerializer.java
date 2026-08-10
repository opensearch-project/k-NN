/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.jni.SimdFp16;

/**
 * Class implements serialization/deserialization of float array as a collection of
 * individual half-precision (FP16) values.
 */
public class KNNVectorAsCollectionOfHalfFloatsSerializer {
    private static final int BYTES_IN_HALF_FLOAT = 2;

    public static final KNNVectorAsCollectionOfHalfFloatsSerializer INSTANCE = new KNNVectorAsCollectionOfHalfFloatsSerializer();

    /**
     * Converts float[] to byte[] using SIMD optimization if supported, otherwise falls back to Java.
     *
     * @param input     the float[] to be serialized into half-precision format
     * @param output    byte[] to fill with the FP16-encoded data
     * @param dimension number of floats to serialize
     */
    public void floatToByteArray(float[] input, byte[] output, int dimension) {
        if (input == null || output == null) {
            throw new IllegalArgumentException("Input/output buffers cannot be null.");
        }
        if (dimension > input.length) {
            throw new IllegalArgumentException("Count exceeds input float array length.");
        }
        if (output.length != input.length * BYTES_IN_HALF_FLOAT) {
            throw new IllegalArgumentException("Output buffer size mismatch. Must be 2x input length.");
        }

        if (SimdFp16.isSIMDSupported()) {
            if (!SimdFp16.encodeFp32ToFp16(input, output, dimension)) {
                throw new IllegalStateException("[KNN] SIMD is supported but native encoding failed unexpectedly.");
            }
            return;
        }

        floatToByteArrayJava(input, output, dimension);
    }

    /**
     * Converts byte[] containing FP16 values to float[].
     * Uses a Java implementation since decode does not have a native SIMD path.
     *
     * @param input     the byte[] containing half-precision encoded data
     * @param output    float[] to fill with the decoded FP32 values
     * @param dimension number of floats to deserialize
     * @param offset    byte offset into the input array where decoding should start
     */
    public float[] byteToFloatArray(BytesRef bytesRef) {
        int dimension = bytesRef.length / BYTES_IN_HALF_FLOAT;
        float[] output = new float[dimension];
        byteToFloatArrayJava(bytesRef.bytes, output, dimension, bytesRef.offset);
        return output;
    }

    public void byteToFloatArray(byte[] input, float[] output, int dimension, int offset) {
        if (input == null || output == null) {
            throw new IllegalArgumentException("Input/output buffers cannot be null.");
        }
        if (offset < 0 || offset + dimension * BYTES_IN_HALF_FLOAT > input.length) {
            throw new IllegalArgumentException("Offset and dimension exceed input length.");
        }

        byteToFloatArrayJava(input, output, dimension, offset);
    }

    public void floatToByteArrayJava(float[] input, byte[] output, int dimension) {
        for (int i = 0; i < dimension; ++i) {
            short fp16 = Float.floatToFloat16(input[i]);
            output[2 * i] = (byte) (fp16 & 0xFF);
            output[2 * i + 1] = (byte) ((fp16 >> 8) & 0xFF);
        }
    }

    /**
     * Java implementation of byte[] FP16 to float[] conversion.
     * Assumes fixed little-endian format matching the encode path.
     */
    private void byteToFloatArrayJava(byte[] input, float[] output, int dimension, int offset) {
        for (int i = offset, j = 0; j < dimension; i += BYTES_IN_HALF_FLOAT, ++j) {
            short fp16 = (short) ((input[i] & 0xFF) | ((input[i + 1] & 0xFF) << 8));
            output[j] = Float.float16ToFloat(fp16);
        }
    }
}
