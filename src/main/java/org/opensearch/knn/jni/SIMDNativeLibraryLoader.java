/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;

import static org.opensearch.knn.index.KNNSettings.isSimdAVX2Disabled;
import static org.opensearch.knn.index.KNNSettings.isSimdAVX512Disabled;
import static org.opensearch.knn.index.KNNSettings.isSimdAVX512SPRDisabled;
import static org.opensearch.knn.jni.PlatformUtils.isSIMDAVX2SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isSIMDAVX512SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isSIMDAVX512SPRSpecSupportedBySystem;

/**
 * Service to interact with SIMD jni layer. Class dependencies should be minimal
 * <p>
 * In order to compile C++ header file, run:
 * javac -h jni/include src/main/java/org/opensearch/knn/jni/SIMDNativeLibraryLoader.java
 *      src/main/java/org/opensearch/knn/common/KNNConstants.java
 */
public class SIMDNativeLibraryLoader {

    // Cached value of SIMD support
    private static final boolean SIMD_SUPPORTED;

    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            try {
                if (!isSimdAVX512SPRDisabled() && isSIMDAVX512SPRSpecSupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX512_SPR_JNI_LIBRARY_NAME);
                } else if (!isSimdAVX512Disabled() && isSIMDAVX512SupportedBySystem()) {
                    System.loadLibrary(KNNConstants.SIMD_AVX512_JNI_LIBRARY_NAME);
                } else if (!isSimdAVX2Disabled() && isSIMDAVX2SupportedBySystem()) {
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
