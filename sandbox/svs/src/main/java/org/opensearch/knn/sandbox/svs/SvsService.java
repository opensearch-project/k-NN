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
 * JNI binding for the isolated Intel SVS native library ({@code libopensearchknn_svs}).
 */
@Log4j2
public class SvsService {

    private static final String SVS_JNI_LIBRARY_NAME = "opensearchknn_svs";

    static {
        KNNLibraryLoader.loadLibraryByVariant(SVS_JNI_LIBRARY_NAME);
        initLibrary();
        KNNEngine.getEngine(SVSConstants.SVS_ENGINE_NAME).setInitialized(true);
        try {
            MergeAbortChecker.isMergeAborted();
            setMergeInterruptCallback();
        } catch (Exception e) {
            log.warn("Unable to add the mergeAbortChecker during SVS initialization", e);
        }
    }

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

    public static native KNNQueryResult[] radiusQueryIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int maxResultWindow
    );

    public static native KNNQueryResult[] radiusQueryIndexWithFilter(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int maxResultWindow,
        long[] filterIds,
        int filterIdsType
    );

    public static native void free(long indexPointer);

    public static native void initLibrary();

    public static native void setMergeInterruptCallback();
}
