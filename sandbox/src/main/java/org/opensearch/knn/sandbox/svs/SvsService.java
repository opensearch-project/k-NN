/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.MergeAbortChecker;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.KNNLibraryLoader;

import java.util.Map;

/**
 * JNI binding for the isolated Intel SVS native library ({@code libopensearchknn_svs}). This is the
 * sandbox tenant's own service: it loads its own native library (which embeds its own SVS-enabled Faiss)
 * and owns the full native lifecycle of {@code svs_vamana} indices, completely separate from the main
 * {@code FaissService}/{@code libopensearchknn_faiss}. It is reached only through {@link SvsEngineProvider}
 * (the {@code SandboxEngineProvider} SPI behind {@code KNNEngine.EXPERIMENTAL}) — main code never references
 * this class.
 *
 * <p>The native surface is deliberately minimal: build iteratively, write, load, top-k query with an
 * optional pre-filter, and free.
 */
@Log4j2
public class SvsService {

    /** Base name of the SVS JNI library; variant suffixes (_avx512_spr, etc.) are appended by the loader. */
    private static final String SVS_JNI_LIBRARY_NAME = "opensearchknn_svs";

    static {
        // System.loadLibrary is centralized in KNNLibraryLoader (enforced by the validateLibraryUsage task).
        KNNLibraryLoader.loadLibraryByVariant(SVS_JNI_LIBRARY_NAME);
        initLibrary();
        // Mark the experimental engine initialized, mirroring how FaissService marks FAISS initialized.
        KNNEngine.EXPERIMENTAL.setInitialized(true);
        try {
            MergeAbortChecker.isMergeAborted();
            setMergeInterruptCallback();
        } catch (Exception e) {
            log.warn("Unable to add the mergeAbortChecker during SVS initialization", e);
        }
    }

    /**
     * Whether this SVS runtime supports LVQ/LeanVec compression (requires Intel AVX-512 SIMD support).
     */
    public static native boolean isLvqLeanvecEnabled();

    public static native long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    public static native void insertToIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);

    public static native void writeIndex(long indexAddress, IndexOutputWithBuffer output);

    public static native long loadIndexWithStream(IndexInputWithBuffer readStream);

    public static native KNNQueryResult[] queryIndex(long indexPointer, float[] queryVector, int k, Map<String, ?> methodParameters);

    public static native KNNQueryResult[] queryIndexWithFilter(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filterIds,
        int filterIdsType
    );

    public static native void free(long indexPointer);

    public static native void initLibrary();

    public static native void setMergeInterruptCallback();
}
