/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import lombok.extern.log4j.Log4j2;
import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
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
     * Loads a JNI library by base name, trying the highest-performance permitted variant first
     * ({@code _avx512_spr}, {@code _avx512}, {@code _avx2}, then the plain base name) and falling back to
     * the next candidate when a variant is not shipped. The suffix scheme mirrors faiss's FAISS_OPT_LEVEL
     * taxonomy, and shipping variants is optional — a single unsuffixed library is fully supported. This
     * class is the only one permitted to call {@link System#loadLibrary}.
     *
     * @param baseLibraryName e.g. {@code opensearchknn_faiss}
     */
    @ExperimentalApi
    public static void loadLibraryByVariant(String baseLibraryName) {
        final List<String> candidates = variantCandidates(baseLibraryName);
        for (int i = 0; i < candidates.size(); i++) {
            try {
                loadLibrary(candidates.get(i));
                return;
            } catch (UnsatisfiedLinkError e) {
                if (i == candidates.size() - 1) {
                    throw e;
                }
                log.info("Library variant unavailable: {}, trying next candidate", candidates.get(i));
            }
        }
    }

    /**
     * Returns the ordered fallback chain of library names for a base name: each SIMD variant the CPU
     * supports and settings have not disabled (widest first), always ending with the plain base name.
     */
    static List<String> variantCandidates(String baseLibraryName) {
        final List<String> candidates = new ArrayList<>();
        if (!isFaissAVX512SPRDisabled() && isAVX512SPRSupportedBySystem()) {
            candidates.add(baseLibraryName + "_avx512_spr");
        }
        if (!isFaissAVX512Disabled() && isAVX512SupportedBySystem()) {
            candidates.add(baseLibraryName + "_avx512");
        }
        if (!isFaissAVX2Disabled() && isAVX2SupportedBySystem()) {
            candidates.add(baseLibraryName + "_avx2");
        }
        candidates.add(baseLibraryName);
        return candidates;
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
