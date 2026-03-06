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
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexFloatFlat;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;

import java.io.IOException;

/**
 * Utility class that performs prefetching of vector data for upcoming node accesses during graph traversal.
 * <p>
 * Prefetching hints the underlying storage to load vector data into memory ahead of time, reducing I/O latency
 * during scoring operations. This is particularly beneficial for memory-optimized (off-heap) search where vector
 * data is read directly from disk-backed index inputs.
 * <p>
 * Supports prefetching for the following {@link KnnVectorValues} implementations:
 * <ul>
 *   <li>{@link FaissIndexFloatFlat.FloatVectorValuesImpl} — FAISS flat float vector storage</li>
 *   <li>{@link FaissIndexBinaryFlat.ByteVectorValuesImpl} — FAISS flat binary vector storage</li>
 *   <li>{@link HasIndexSlice} — Lucene native vector storage backed by a sliced index input</li>
 * </ul>
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
        switch (vectorValues) {
            case FaissIndexFloatFlat.FloatVectorValuesImpl floatImpl:
                floatImpl.prefetch(nodes, numNodes);
                break;
            case FaissIndexBinaryFlat.ByteVectorValuesImpl binaryImpl:
                binaryImpl.prefetch(nodes, numNodes);
                break;
            case HasIndexSlice luceneKNNVectorValues:
                // Since Lucene uses HasIndexSlice, we can use the slice to prefetch and for lucene base offset for
                // sliced index input is always 0.
                PrefetchHelper.prefetch(luceneKNNVectorValues.getSlice(), 0, vectorValues.getVectorByteLength(), nodes, numNodes);
                break;
            default:
                log.warn("Not able to do prefetch on instance {}", vectorValues.getClass().getSimpleName());
        }
    }

}
