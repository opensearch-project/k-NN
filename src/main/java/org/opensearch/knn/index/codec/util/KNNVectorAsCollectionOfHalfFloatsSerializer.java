/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.opensearch.knn.jni.SIMDDecoding;
import org.opensearch.knn.jni.SIMDEncoding;

import java.nio.ByteOrder;

/**
 * Class implements KNNVectorSerializer based on serialization/deserialization of float array
 * as a collection of individual half-precision values
 */
public class KNNVectorAsCollectionOfHalfFloatsSerializer {
    private static final int BYTES_IN_HALF_FLOAT = 2;
    private static final boolean IS_LITTLE_ENDIAN = ByteOrder.nativeOrder().equals(ByteOrder.LITTLE_ENDIAN);

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

        if (SIMDEncoding.isSIMDSupported() && SIMDEncoding.convertFP32ToFP16(input, output, dimension)) {
            return;
        }

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
                    "Invalid byte array length (" + input.length +
                            "). Must be a multiple of " + BYTES_IN_HALF_FLOAT + " to represent float16 values.");
        }
        if (offset < 0 || offset + dimension * BYTES_IN_HALF_FLOAT > input.length) {
            throw new IllegalArgumentException("Offset and dimension exceed input length.");
        }

        if (SIMDDecoding.isSIMDSupported() && SIMDDecoding.convertFP16ToFP32(input, output, dimension, offset)) {
            return;
        }

        byteToFloatArrayFallback(input, output, dimension, offset);
    }

    /**
     * Fallback Java implementation of float[] to byte[] conversion.
     * Handles both little-endian and big-endian platforms.
     *
     * @param input  float array to convert
     * @param output output byte array to hold FP16-encoded values
     * @param dimension number of elements to convert
     */
    private void floatToByteArrayFallback(float[] input, byte[] output, int dimension) {
        if (IS_LITTLE_ENDIAN) {
            for (int i = 0; i < dimension; ++i) {
                short fp16 = Float.floatToFloat16(input[i]);
                output[2 * i] = (byte) (fp16 & 0xFF); // low byte
                output[2 * i + 1] = (byte) ((fp16 >> 8) & 0xFF); // high byte
            }
        } else {
            for (int i = 0; i < dimension; ++i) {
                short fp16 = Float.floatToFloat16(input[i]);
                output[2 * i] = (byte) ((fp16 >> 8) & 0xFF); // high byte
                output[2 * i + 1] = (byte) (fp16 & 0xFF); // low byte
            }
        }
    }

    /**
     * Fallback Java implementation of byte[] to float[] conversion.
     * Handles both little-endian and big-endian platforms.
     *
     * @param input  byte array containing FP16-encoded data
     * @param output output float array
     * @param dimension number of floats to decode
     * @param offset offset into the input byte array
     */
    private void byteToFloatArrayFallback(byte[] input, float[] output, int dimension, int offset) {
        if (IS_LITTLE_ENDIAN) {
            for (int i = offset, j = 0; j < dimension; i += 2, ++j) {
                final short fp16 = (short) ((input[i] & 0xFF) | ((input[i + 1] & 0xFF) << 8));
                output[j] = Float.float16ToFloat(fp16);
            }
        } else {
            for (int i = offset, j = 0; j < dimension; i += 2, ++j) {
                final short fp16 = (short) (((input[i] & 0xFF) << 8) | (input[i + 1] & 0xFF));
                output[j] = Float.float16ToFloat(fp16);
            }
        }
    }
}
