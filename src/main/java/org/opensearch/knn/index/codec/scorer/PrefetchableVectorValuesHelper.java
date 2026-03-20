/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.KnnVectorValues;
import java.io.IOException;

/**
 * Utility class that performs prefetching of vector data for upcoming node accesses during graph traversal.
 * <p>
 * Prefetching hints the underlying storage to load vector data into memory ahead of time, reducing I/O latency
 * during scoring operations. This is particularly beneficial for memory-optimized (off-heap) search where vector
 * data is read directly from disk-backed index inputs.
 * <p>
 * Supports prefetching for any {@link KnnVectorValues} implementation that also implements {@link HasIndexSlice},
 * which provides access to the underlying sliced index input for prefetch operations.
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
class PrefetchableVectorValuesHelper {

    /**
     * Attempts to prefetch vector data for the given node ordinals.
     * <p>
     * If the provided {@link KnnVectorValues} implementation supports prefetching, this method will issue
     * prefetch hints for the specified nodes. Otherwise, it logs an informational message and returns without
     * performing any prefetch.
     *
     * @param vectorValues the vector values instance to prefetch from
     * @param nodes        array of node ordinals whose vector data should be prefetched
     * @param numNodes     number of valid entries in the {@code nodes} array to prefetch
     * @throws IOException if an I/O error occurs during prefetching
     */
    public static void mayBeDoPrefetch(final KnnVectorValues vectorValues, final int[] nodes, final int numNodes) throws IOException {
        if (vectorValues instanceof HasIndexSlice vectorValuesWithSlice) {
            // passing base offset as 0, since the index input is a slice and its base offset is 0.
            PrefetchHelper.prefetch(vectorValuesWithSlice.getSlice(), 0, vectorValues.getVectorByteLength(), nodes, numNodes);
        } else {
            log.warn("Not able to do prefetch on instance {}", vectorValues.getClass().getSimpleName());
        }
    }

}
