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

public class SIMDEncoding {

    static {
        // Ensures native library is loaded as soon as class is referenced
        SIMDNativeLibraryLoader.isSIMDSupported();
    }

    /**
     * Returns whether native SIMD encoding is available.
     * @return true if SIMD encoding is supported, false otherwise
     */
    public static boolean isSIMDSupported() {
        return SIMDNativeLibraryLoader.isSIMDSupported();
    }

    /**
     * Converts an array of float values to half-precision (fp16) bytes using native code.
     *
     * @param input float array containing float32 values
     * @param output byte array to fill with the converted half-float values (2 bytes per value)
     * @param count number of float values to convert
     * @return true if native decoding succeeded (including alignment), false if fallback is required
     */
    public static native boolean convertFP32ToFP16(float[] input, byte[] output, int count);
}
