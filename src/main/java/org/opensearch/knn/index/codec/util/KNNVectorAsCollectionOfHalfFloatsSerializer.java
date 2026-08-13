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
        if (dimension < 0 || dimension > input.length) {
            throw new IllegalArgumentException(
                "Dimension must be non-negative and not exceed input float array length. dimension="
                    + dimension
                    + ", input.length="
                    + input.length
            );
        }
        if (output.length != dimension * BYTES_IN_HALF_FLOAT) {
            throw new IllegalArgumentException(
                "Output buffer size mismatch. Must be 2x dimension. output.length=" + output.length + ", dimension=" + dimension
            );
        }

        if (SimdFp16.isSIMDSupported()) {
            if (SimdFp16.encodeFp32ToFp16(input, output, dimension) == false) {
                throw new IllegalStateException("[KNN] SIMD is supported but native encoding failed unexpectedly.");
            }
            return;
        }

        floatToByteArrayJava(input, output, dimension);
    }

    /**
     * Java implementation of float[] to FP16 byte[] conversion.
     * Only reached via {@link #floatToByteArray}, which validates its arguments first.
     */
    private void floatToByteArrayJava(float[] input, byte[] output, int dimension) {
        for (int i = 0; i < dimension; ++i) {
            short fp16 = Float.floatToFloat16(input[i]);
            output[2 * i] = (byte) (fp16 & 0xFF);
            output[2 * i + 1] = (byte) ((fp16 >> 8) & 0xFF);
        }
    }

    /**
     * Converts a {@link BytesRef} containing FP16 values to float[].
     *
     * @param bytesRef the BytesRef containing half-precision encoded data
     */
    public float[] byteToFloatArray(BytesRef bytesRef) {
        if (bytesRef == null || bytesRef.bytes == null) {
            throw new IllegalArgumentException("BytesRef and its backing array cannot be null.");
        }
        if (bytesRef.length < 0 || bytesRef.length % BYTES_IN_HALF_FLOAT != 0) {
            throw new IllegalArgumentException(
                "BytesRef length must be non-negative and a multiple of " + BYTES_IN_HALF_FLOAT + ". bytesRef.length=" + bytesRef.length
            );
        }
        if (bytesRef.offset < 0 || bytesRef.offset + bytesRef.length > bytesRef.bytes.length) {
            throw new IllegalArgumentException(
                "BytesRef offset and length exceed backing array length. bytesRef.offset="
                    + bytesRef.offset
                    + ", bytesRef.length="
                    + bytesRef.length
                    + ", bytesRef.bytes.length="
                    + bytesRef.bytes.length
            );
        }
        int dimension = bytesRef.length / BYTES_IN_HALF_FLOAT;
        float[] output = new float[dimension];
        byteToFloatArray(bytesRef.bytes, output, dimension, bytesRef.offset);
        return output;
    }

    /**
     * Converts byte[] containing FP16 values to float[] using SIMD optimization if supported,
     * otherwise falls back to Java.
     *
     * @param input     the byte[] containing half-precision encoded data
     * @param output    float[] to fill with the decoded FP32 values
     * @param dimension number of floats to deserialize
     * @param offset    byte offset into the input array where decoding should start
     */
    public void byteToFloatArray(byte[] input, float[] output, int dimension, int offset) {
        if (input == null || output == null) {
            throw new IllegalArgumentException("Input/output buffers cannot be null.");
        }
        if (dimension < 0) {
            throw new IllegalArgumentException("Dimension cannot be negative. dimension=" + dimension);
        }
        if (offset < 0 || offset + dimension * BYTES_IN_HALF_FLOAT > input.length) {
            throw new IllegalArgumentException(
                "Offset and dimension exceed input length. offset=" + offset + ", dimension=" + dimension + ", input.length=" + input.length
            );
        }
        if (output.length < dimension) {
            throw new IllegalArgumentException(
                "Output buffer too small for dimension. output.length=" + output.length + ", dimension=" + dimension
            );
        }

        if (SimdFp16.isSIMDSupported()) {
            if (!SimdFp16.decodeFp16ToFp32(input, offset, output, dimension)) {
                throw new IllegalStateException("[KNN] SIMD is supported but native decoding failed unexpectedly.");
            }
            return;
        }

        byteToFloatArrayJava(input, output, dimension, offset);
    }

    /**
     * Java implementation of FP16 byte[] to float[] conversion.
     * Only reached via {@link #byteToFloatArray}, which validates its arguments first.
     */
    private void byteToFloatArrayJava(byte[] input, float[] output, int dimension, int offset) {
        for (int i = offset, j = 0; j < dimension; i += BYTES_IN_HALF_FLOAT, ++j) {
            short fp16 = (short) ((input[i] & 0xFF) | ((input[i + 1] & 0xFF) << 8));
            output[j] = Float.float16ToFloat(fp16);
        }
    }
}
