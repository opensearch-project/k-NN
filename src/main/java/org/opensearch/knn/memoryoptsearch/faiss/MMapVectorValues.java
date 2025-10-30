/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

/**
 * Provides access to memory-mapped vector data divided into contiguous chunks.
 * <p>
 * Each chunk represents a continuous region of memory storing vectors. The chunks
 * are described as pairs of values within a {@code long[]} array.
 * For each chunk {@code j}, the element at index {@code 2 * j} represents the
 * starting memory address, and the element at index {@code 2 * j + 1} represents
 * the size of that chunk in bytes.
 * <p>
 * For example:
 * <ul>
 *   <li>{@code addressAndSize[6]} is the starting address of the third chunk</li>
 *   <li>{@code addressAndSize[7]} is the size (in bytes) of the third chunk</li>
 * </ul>
 */
public interface MMapVectorValues {
    /**
     * Returns an array describing the memory-mapped vector chunks.
     * <p>
     * Each pair of consecutive elements corresponds to one chunk:
     * <ul>
     *   <li>{@code addressAndSize[2 * j]} — the starting memory address of the j-th chunk</li>
     *   <li>{@code addressAndSize[2 * j + 1]} — the size of that chunk in bytes</li>
     * </ul>
     *
     * @return a {@code long[]} array containing address–size pairs for each memory chunk
     */
    long[] getAddressAndSize();
}
