/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

import java.util.Map;

/**
 * Generic native-index lifecycle contract for an engine that is contributed at runtime (rather than being a
 * built-in such as Faiss or NMSLIB). {@link org.opensearch.knn.jni.JNIService} routes the eight lifecycle/search operations
 * (init/insert/write/template/load/query/radiusQuery/free) for the registered engine to an implementation of
 * this interface, so it can drive its own isolated JNI library without {@code JNIService} holding any
 * compile-time reference to that engine; binary indexes, training and shared index state remain core-only today.
 *
 * <p>The method set mirrors {@link org.opensearch.knn.jni.JNIService}'s per-engine entry points. An implementation that does not
 * support a particular operation (for example template-based builds or radial search) should throw
 * {@link UnsupportedOperationException}; the corresponding capability checks in the core typically keep those
 * paths unreachable, so the throws are defensive backstops.
 *
 * <p>Implementations must be thread-safe: {@link org.opensearch.knn.jni.JNIService}'s static methods are invoked concurrently from
 * search and merge threads. The service instance is created eagerly at engine discovery, but it must defer any
 * native library loading to first use (via {@link org.opensearch.knn.jni.KNNLibraryLoader#loadLibraryByVariant(String)}).
 */
@ExperimentalApi
public interface NativeEngineService {

    long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    /** {@code vectorsAddress} is an off-heap address of the vectors to copy. */
    void insertToIndex(int[] docs, long vectorsAddress, int dimension, Map<String, Object> parameters, long indexAddress);

    void writeIndex(IndexOutputWithBuffer output, long indexAddress, Map<String, Object> parameters, boolean skipFlat);

    void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        int dim,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        Map<String, Object> parameters
    );

    long loadIndex(IndexInputWithBuffer readStream, Map<String, Object> parameters);

    /** {@code filterIdsType} discriminates the {@code filteredIds} encoding. */
    KNNQueryResult[] queryIndex(
        long indexPointer,
        float[] queryVector,
        int k,
        Map<String, ?> methodParameters,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    );

    KNNQueryResult[] radiusQueryIndex(
        long indexPointer,
        float[] queryVector,
        float radius,
        Map<String, ?> methodParameters,
        int indexMaxResultWindow,
        long[] filteredIds,
        int filterIdsType,
        int[] parentIds
    );

    void free(long indexPointer, boolean isBinaryIndex);
}
