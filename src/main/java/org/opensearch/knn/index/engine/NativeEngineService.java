/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;

/**
 * Native-index lifecycle of a runtime-registered engine. {@link org.opensearch.knn.jni.JNIService} routes its
 * per-engine operations here, so the engine drives its own JNI library with no compile-time reference in core;
 * binary indexes, training and shared index state remain core-only. An unsupported operation should throw
 * {@link UnsupportedOperationException}; core capability checks normally keep it unreachable. Pointer-typed
 * longs are engine-defined opaque handles.
 *
 * <p>Implementations must be thread-safe (search and merge threads call concurrently) and must defer native
 * library loading to first use. Signatures change between experimental iterations; a jar compiled against an
 * older shape fails with {@link UnsupportedOperationException}, not a linkage error.
 */
@ExperimentalApi
public interface NativeEngineService {

    long initIndex(NativeIndexBuildParams params);

    /** {@code vectorsAddress} is an off-heap address of the vectors to copy. */
    void insertToIndex(int[] docs, long vectorsAddress, long indexAddress, NativeIndexBuildParams params);

    void writeIndex(IndexOutputWithBuffer output, long indexAddress, NativeIndexBuildParams params);

    void createIndexFromTemplate(
        int[] ids,
        long vectorsAddress,
        IndexOutputWithBuffer output,
        byte[] templateIndex,
        NativeIndexBuildParams params
    );

    long loadIndex(IndexInputWithBuffer readStream, NativeIndexBuildParams params);

    KNNQueryResult[] queryIndex(long indexPointer, NativeSearchParams params);

    KNNQueryResult[] radiusQueryIndex(long indexPointer, NativeSearchParams params);

    void free(long indexPointer);
}
