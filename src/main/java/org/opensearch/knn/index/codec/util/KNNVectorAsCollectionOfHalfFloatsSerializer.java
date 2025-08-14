/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.jni.SimdFp16;

/**
 * Class implements KNNVectorSerializer based on serialization/deserialization of float array
 * as a collection of individual half-precision values
 */
@Log4j2
public class KNNVectorAsCollectionOfHalfFloatsSerializer {
    private static final int BYTES_IN_HALF_FLOAT = 2;

    public static final KNNVectorAsCollectionOfHalfFloatsSerializer INSTANCE = new KNNVectorAsCollectionOfHalfFloatsSerializer();

    /**
     * Converts float[] to byte[] using SIMD optimization if supported, otherwise fallback to Java.
     * @param input the float[] to be serialized into half-precision format
     * @param output byte[] containing the float16-encoded data
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

        log.warn("SIMD FP32 to FP16 encoding not supported on this platform, falling back to Java implementation.");
        floatToByteArrayFallback(input, output, dimension);
    }

    /**
     * Converts byte[] to float[] using SIMD optimization if supported, otherwise fallback to Java.
     * @param input the byte[] containing half-precision encoded data
     * @param output a float[] containing the decoded float32 values
     * @param dimension number of floats to deserialize
     * @param offset offset in the output array where deserialization should start
     */
    public void byteToFloatArray(byte[] input, float[] output, int dimension, int offset) {
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
        if (offset < 0 || offset + dimension * BYTES_IN_HALF_FLOAT > input.length) {
            throw new IllegalArgumentException("Offset and dimension exceed input length.");
        }

        if (SimdFp16.isSIMDSupported()) {
            if (!SimdFp16.decodeFp16ToFp32(input, output, dimension, offset)) {
                throw new IllegalStateException("[KNN] SIMD is supported but native decoding failed unexpectedly.");
            }
            return;
        }

        log.warn("SIMD FP16 to FP32 decoding not supported on this platform, falling back to Java implementation.");
        byteToFloatArrayFallback(input, output, dimension, offset);
    }

    /**
     * Fallback Java implementation of float[] to byte[] conversion.
     * Since the format is fully controlled, a fixed little-endian format is
     * chosen for consistency, as it is standard on modern CPUs and ML formats.
     * Fixed byte order ensures portability and consistent encoding across platforms.
     *
     * @param input  float array to convert
     * @param output output byte array to hold FP16-encoded values
     * @param dimension number of elements to convert
     */
    private void floatToByteArrayFallback(float[] input, byte[] output, int dimension) {
        for (int i = 0; i < dimension; ++i) {
            short fp16 = Float.floatToFloat16(input[i]);
            output[2 * i] = (byte) (fp16 & 0xFF); // low byte
            output[2 * i + 1] = (byte) ((fp16 >> 8) & 0xFF); // high byte
        }
    }

    /**
     * Fallback Java implementation of byte[] to float[] conversion.
     * Assumes the input byte array is in fixed little-endian format, matching the write path.
     * Fixed byte order ensures portability and consistent decoding across platforms.
     *
     * @param input  byte array containing FP16-encoded data
     * @param output output float array
     * @param dimension number of floats to decode
     * @param offset offset into the input byte array
     */
    private void byteToFloatArrayFallback(byte[] input, float[] output, int dimension, int offset) {
        for (int i = offset, j = 0; j < dimension; i += 2, ++j) {
            final short fp16 = (short) ((input[i] & 0xFF) | ((input[i + 1] & 0xFF) << 8));
            output[j] = Float.float16ToFloat(fp16);
        }
    }
}
