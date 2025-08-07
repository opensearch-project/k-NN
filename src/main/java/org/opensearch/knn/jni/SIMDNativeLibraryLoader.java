/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;

public class SIMDNativeLibraryLoader {

    // Cached value of SIMD support
    private static final boolean SIMD_SUPPORTED;

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            try {
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
}
