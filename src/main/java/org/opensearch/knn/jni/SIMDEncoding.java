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


import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;

public class SIMDEncoding {

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            try {
                // Load SIMD library based on support
                if (PlatformUtils.isAVX512SPRSupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX512_SPR_JNI_LIBRARY_NAME);
                } else if (PlatformUtils.isAVX512SupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX512_JNI_LIBRARY_NAME);
                } else if (PlatformUtils.isAVX2SupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX2_JNI_LIBRARY_NAME);
                } else {
                    System.loadLibrary(KNNConstants.SIMD_JNI_LIBRARY_NAME);
                }
            } catch (UnsatisfiedLinkError e) {
                throw new RuntimeException("[KNN] Failed to load native SIMD library", e);
            }
            return null;
        });

        // Cache native SIMD support check once during class loading
        SIMD_SUPPORTED = isSIMDSupportedNative();
    }

    // Cached value of SIMD support
    private static final boolean SIMD_SUPPORTED;

    /**
     * Cached check for whether SIMD encoding is supported.
     *
     * @return true if native SIMD is supported and enabled, false otherwise
     */
    public static boolean isSIMDSupported() {
        return SIMD_SUPPORTED;
    }

    /**
     * Actual JNI native call to check if the platform supports SIMD-based FP16 encoding.
     *
     * @return true if SIMD is supported and enabled, false otherwise
     */
    private static native boolean isSIMDSupportedNative();

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