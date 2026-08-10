/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

/**
 * A service that provides SIMD-accelerated FP32-to-FP16 encoding via native code.
 * The appropriate native library variant is selected and loaded based on system CPU capabilities.
 */
public class SimdFp16 {

    private static final boolean SIMD_SUPPORTED;

    static {
        KNNLibraryLoader.loadSimdLibrary();
        SIMD_SUPPORTED = isSIMDSupportedNative();
    }

    /**
     * Cached check for whether SIMD FP16 encoding is supported on this platform.
     *
     * @return true if native SIMD is supported and enabled, false otherwise
     */
    public static boolean isSIMDSupported() {
        return SIMD_SUPPORTED;
    }

    /**
     * Checks if the native library supports SIMD-based FP16 encoding.
     *
     * @return true if SIMD is supported and enabled, false otherwise
     */
    private static native boolean isSIMDSupportedNative();

    /**
     * Converts an array of float32 values to half-precision (FP16) bytes using native SIMD code.
     *
     * @param input  float array containing FP32 values
     * @param output byte array to fill with the converted FP16 values (2 bytes per value)
     * @param count  number of float values to convert
     * @return true if native encoding succeeded, false if fallback is required
     */
    public static native boolean encodeFp32ToFp16(float[] input, byte[] output, int count);
}
