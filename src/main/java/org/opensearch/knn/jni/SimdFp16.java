/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;

import static org.opensearch.knn.index.KNNSettings.isAVX2Disabled;
import static org.opensearch.knn.index.KNNSettings.isAVX512Disabled;
import static org.opensearch.knn.index.KNNSettings.isAVX512SPRDisabled;
import static org.opensearch.knn.jni.PlatformUtils.isSIMDAVX2SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isSIMDAVX512SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isSIMDAVX512SPRSupportedBySystem;

/**
 * Service to interact with SIMD jni layer. Class dependencies should be minimal
 * <p>
 * In order to compile C++ header file, run:
 * javac -h jni/include src/main/java/org/opensearch/knn/jni/SimdFp16.java
 *      src/main/java/org/opensearch/knn/common/KNNConstants.java
 */
public class SimdFp16 {

    // Cached value of SIMD support
    private static final boolean SIMD_SUPPORTED;

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            try {
                if (!isAVX512SPRDisabled() && isSIMDAVX512SPRSupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX512_SPR_JNI_LIBRARY_NAME);
                } else if (!isAVX512Disabled() && isSIMDAVX512SupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX512_JNI_LIBRARY_NAME);
                } else if (!isAVX2Disabled() && isSIMDAVX2SupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX2_JNI_LIBRARY_NAME);
                } else {
                    System.loadLibrary(KNNConstants.SIMD_JNI_LIBRARY_NAME);
                }
            } catch (UnsatisfiedLinkError e) {
                throw new RuntimeException("[KNN] Failed to load native SIMD library", e);
            }
            return null;
        });

        SIMD_SUPPORTED = isSIMDSupportedNative();
    }

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
    public static native boolean encodeFp32ToFp16(float[] input, byte[] output, int count);

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
    public static native boolean decodeFp16ToFp32(byte[] input, float[] output, int count, int offset);
}
