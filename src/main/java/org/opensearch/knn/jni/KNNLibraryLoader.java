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
     */
    static void loadFaissLibrary() {
        loadLibraryByVariant(KNNConstants.FAISS_JNI_LIBRARY_NAME);
    }

    /**
     * Loads the highest-performance available variant of a JNI library given its base name, selecting
     * AVX512-SPR, then AVX512, then AVX2, then the plain library — each taken only when the system supports
     * it and the corresponding {@code knn.faiss.avx*.disabled} setting has not disabled it (those settings
     * govern SIMD selection for every variant-built library, not just Faiss). Backs the built-in Faiss and
     * SIMD libraries, and is public so an experimental {@code :sandbox} tenant can load its own native
     * library through this class (the only place permitted to call {@link System#loadLibrary}); the base
     * name is supplied by the tenant, so no tenant-specific name lives here.
     *
     * @param baseLibraryName e.g. {@code opensearchknn_faiss}; variant suffixes ({@code _avx512_spr} etc.) are appended.
     */
    public static void loadLibraryByVariant(String baseLibraryName) {
        if (!isFaissAVX512SPRDisabled() && isAVX512SPRSupportedBySystem()) {
            loadLibrary(baseLibraryName + "_avx512_spr");
        } else if (!isFaissAVX512Disabled() && isAVX512SupportedBySystem()) {
            loadLibrary(baseLibraryName + "_avx512");
        } else if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
            loadLibrary(baseLibraryName + "_avx2");
        } else {
            loadLibrary(baseLibraryName);
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
     */
    static void loadSimdLibrary() {
        loadLibraryByVariant(KNNConstants.DEFAULT_SIMD_COMPUTING_JNI_LIBRARY_NAME);
    }
}
