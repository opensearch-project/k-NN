/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.jni;

public class SIMDDecoding {

    /**
     * Returns whether native SIMD decoding is available.
     * This reuses the cached value from SIMDEncoding.
     * @return true if SIMD decoding is supported, false otherwise
     */
    public static boolean isSIMDSupported() {
        return SIMDEncoding.isSIMDSupported();
    }

    /**
     * Converts a byte array containing float16 (half-precision) values into a float array of float32 values
     * using native code. Uses SIMD acceleration if available, or falls back to Java if JNI returns failure.
     *
     * @param input byte array containing FP16-encoded values
     * @param output float array to write the decoded FP32 values
     * @param count number of float values to decode
     * @param offset byte offset into the input array to begin decoding
     * @return true if native decoding succeeded (including alignment), false if fallback is required
     */
    public static native boolean convertFP16ToFP32(byte[] input, float[] output, int count, int offset);
}
