/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;

import java.io.IOException;
import java.util.Arrays;

/**
 * Helper class to prefetch vector data from disk to improve query performance.
 * <p>
 * Prefetches vector data using exact byte range strategy. Vectors are grouped together when
 * the end of a vector (start offset + vector size) falls within 128KB from the group's start offset.
 * This minimizes I/O operations while fetching only the exact bytes needed.
 * <p>
 * Prefetching is controlled by {@link KNNFeatureFlags#isPrefetchEnabled()}. When disabled,
 * no prefetch operations are performed.
 * <p>
 * <b>Example:</b><br>
 * Given vectors at offsets [100KB, 120KB, 300KB] with 10KB vector size:
 * <ul>
 *   <li>Group 1: Prefetch 40KB (100KB to 130KB, covers vectors at 100KB and 120KB)</li>
 *   <li>Group 2: Prefetch 10KB (300KB to 310KB, covers vector at 300KB)</li>
 * </ul>
 * Result: 2 prefetch operations, 50KB total prefetched (vs 6 random seeks without prefetch)
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class PrefetchHelper {

    // TODO: If needed we can get this value via Cluster Settings
    private static final long BYTES_128 = 128 * 1024;

    /**
     * Prefetches vector data from disk using exact byte range strategy.
     * <p>
     * Vectors are sorted by offset and grouped when the end of a vector falls within 128KB
     * from the group start. Each group is prefetched with exact byte range needed.
     * Returns early if prefetch is disabled, ordinals are null, or fewer than 2 vectors.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    public static void prefetch(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        if (ordsToPrefetch == null || numOrds <= 1) {
            return;
        }
        if (KNNFeatureFlags.isPrefetchEnabled()) {
            prefetchExactVectorSize(indexInput, baseOffset, oneVectorByteSize, ordsToPrefetch, numOrds);
        } else {
            log.debug("KNNVectors Prefetch is disabled");
        }
    }

    /**
     * Prefetches vectors using exact byte ranges.
     * <p>
     * Groups vectors within 128KB ranges and prefetches only the exact bytes needed for each group.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    private static void prefetchExactVectorSize(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        Arrays.sort(ordsToPrefetch, 0, numOrds);

        int groupCount = 1;
        long groupStartOffset = baseOffset + (long) ordsToPrefetch[0] * oneVectorByteSize;

        for (int i = 1; i < numOrds; i++) {
            long currentOffset = baseOffset + (long) ordsToPrefetch[i] * oneVectorByteSize;
            if ((currentOffset + oneVectorByteSize) - groupStartOffset > BYTES_128) {
                long prevOffset = baseOffset + (long) ordsToPrefetch[i - 1] * oneVectorByteSize;
                indexInput.prefetch(groupStartOffset, (prevOffset + oneVectorByteSize) - groupStartOffset);
                groupCount++;
                groupStartOffset = currentOffset;
            }
        }
        // Prefetch final group
        long lastOffset = baseOffset + (long) ordsToPrefetch[numOrds - 1] * oneVectorByteSize;
        indexInput.prefetch(groupStartOffset, (lastOffset + oneVectorByteSize) - groupStartOffset);

        log.trace("Prefetching grouped [{}] vectors where num of ords was [{}] using exact prefetch size", groupCount, numOrds);
    }
}
