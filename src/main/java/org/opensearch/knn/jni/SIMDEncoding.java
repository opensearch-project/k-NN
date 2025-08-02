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

import static org.opensearch.knn.index.KNNSettings.*;
import static org.opensearch.knn.jni.PlatformUtils.*;

public class SIMDEncoding {

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            if (isAVX512SPRSupportedBySystem()) {
                System.loadLibrary(KNNConstants.SIMD_AVX512_SPR_JNI_LIBRARY_NAME);
            } else if (isAVX512SupportedBySystem()) {
                System.loadLibrary(KNNConstants.SIMD_AVX512_JNI_LIBRARY_NAME);
            } else if (isAVX2SupportedBySystem()) {
                System.loadLibrary(KNNConstants.SIMD_AVX2_JNI_LIBRARY_NAME);
            } else {
                System.loadLibrary(KNNConstants.SIMD_JNI_LIBRARY_NAME);
            }

            return null;
        });
    }

    /**
     * Checks if the platform supports SIMD-based FP16 encoding.
     *
     * @return true if SIMD is supported and enabled, false otherwise
     */
    public static native boolean isSIMDSupported();

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