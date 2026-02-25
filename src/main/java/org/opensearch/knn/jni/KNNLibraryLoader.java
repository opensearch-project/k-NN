/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.HashSet;
import java.util.Set;

import static org.opensearch.knn.index.KNNSettings.isFaissAVX2Disabled;
import static org.opensearch.knn.index.KNNSettings.isFaissAVX512Disabled;
import static org.opensearch.knn.index.KNNSettings.isFaissAVX512SPRDisabled;
import static org.opensearch.knn.jni.PlatformUtils.isAVX2SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SPRSupportedBySystem;

/**
 * Thread-safe loader for KNN native libraries.
 *
 * Ensures each library is loaded exactly once across all threads using synchronized loading
 * and tracking of loaded libraries. All library loading is performed with appropriate
 * security privileges.
 *
 * Note: this is the only class that is allowed to load libraries, and all non-private
 * methods are automatically tested.
 */
@Log4j2
public class KNNLibraryLoader {
    /** Set of already loaded library names to prevent duplicate loading */
    static protected Set<String> loaded = new HashSet<>();
    /** Lock object for synchronizing library loading operations */
    static final Object lock = new Object();

    /**
     * Thread-safe library loading with duplicate prevention.
     *
     * @param name the library name to load
     */
    static private void loadLibrary(String name) {
        synchronized (lock) {
            if (loaded.contains(name)) {
                log.info("Library already loaded: {}", name);
                return;
            }
            try {
                AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                    System.loadLibrary(name);
                    return null;
                });
                loaded.add(name);
            } catch (UnsatisfiedLinkError e) {
                log.error("Failed to load library: {}", name);
                throw e;
            }
            log.info("Loaded library: {}", name);
        }
    }

    /**
     * Loads the appropriate Faiss library based on system capabilities and settings.
     *
     * Selects the highest performance variant available:
     * 1. AVX512 SPR if supported and not disabled
     * 2. AVX512 if supported and not disabled
     * 3. AVX2 if supported and not disabled
     * 4. Default fallback library
     */
    static void loadFaissLibrary() {
        if (!isFaissAVX512SPRDisabled() && isAVX512SPRSupportedBySystem()) {
            loadLibrary(KNNConstants.FAISS_AVX512_SPR_JNI_LIBRARY_NAME);
        } else if (!isFaissAVX512Disabled() && isAVX512SupportedBySystem()) {
            loadLibrary(KNNConstants.FAISS_AVX512_JNI_LIBRARY_NAME);
        } else if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
            loadLibrary(KNNConstants.FAISS_AVX2_JNI_LIBRARY_NAME);
        } else {
            loadLibrary(KNNConstants.FAISS_JNI_LIBRARY_NAME);
        }
    }

    /**
     * Loads the NMSLIB JNI library for nearest neighbor search operations.
     */
    static void loadNmslibLibrary() {
        loadLibrary(KNNConstants.NMSLIB_JNI_LIBRARY_NAME);
    }

    /**
     * Loads the common JNI library containing shared functionality.
     */
    static void loadCommonLibrary() {
        loadLibrary(KNNConstants.COMMON_JNI_LIBRARY_NAME);
    }

    /**
     * Loads the appropriate SIMD computing library based on system capabilities.
     *
     * Follows the same selection logic as Faiss library:
     * 1. AVX512 SPR variant if supported and not disabled
     * 2. AVX512 variant if supported and not disabled
     * 3. AVX2 variant if supported and not disabled
     * 4. Default variant as fallback
     */
    static void loadSimdLibrary() {
        if (!isFaissAVX512SPRDisabled() && isAVX512SPRSupportedBySystem()) {
            loadLibrary(KNNConstants.SIMD_COMPUTING_AVX512_SPR_JNI_LIBRARY_NAME);
        } else if (!isFaissAVX512Disabled() && isAVX512SupportedBySystem()) {
            loadLibrary(KNNConstants.SIMD_COMPUTING_AVX512_JNI_LIBRARY_NAME);
        } else if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
            loadLibrary(KNNConstants.SIMD_COMPUTING_AVX2_JNI_LIBRARY_NAME);
        } else {
            loadLibrary(KNNConstants.DEFAULT_SIMD_COMPUTING_JNI_LIBRARY_NAME);
        }
    }
}
