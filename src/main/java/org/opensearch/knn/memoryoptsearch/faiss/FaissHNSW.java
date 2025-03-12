/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * While it follows the same steps as the original FAISS deserialization, differences in how the JVM and C++ handle floating-point
 * calculations can lead to slight variations in results. However, such cases are very rare, and in most instances, the results are
 * identical to FAISS. Even when there are ranking differences, they do not impact the precision or recall of the search.
 * For more details, refer to the [FAISS HNSW implementation](
 * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.h">...</a>).
 */
@Getter
public class FaissHNSW {
    // Cumulative number of neighbors per each level.
    private int[] cumNumberNeighborPerLevel;
    // offsets[i]:offset[i+1] gives all the neighbors for vector i
    // Offset to be added to cumNumberNeighborPerLevel[level] to get the actual start offset of neighbor list.
    private long[] offsets = null;
    // Neighbor list storage.
    private FaissSection neighbors;
    // levels[i] = the maximum levels of `i`th vector + 1.
    // Ex: If 544th vector has three levels (e.g. 0-level, 1-level, 2-level), then levels[433] would be 3.
    // This indicates that 544th vector exists at all levels of (0-level, 1-level, 2-level).
    private FaissSection levels;
    // Entry point in HNSW graph
    private int entryPoint;
    // Maximum level of HNSW graph
    private int maxLevel = -1;
    // Default efSearch parameter. This determines the navigation queue size.
    // More value, algorithm will more navigate candidates.
    private int efSearch = 16;
    // Total number of vectors stored in graph.
    private long totalNumberOfVectors;

    /**
     * Partially loads the FAISS HNSW graph from the provided index input stream.
     * The graph is divided into multiple sections, and this method marks the starting offset of each section then skip to the next
     * section instead of loading the entire graph into memory. During the search, bytes will be accessed via {@link IndexInput}.
     *
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param totalNumberOfVectors The total number of vectors stored in the graph.
     *
     * FYI <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L522">FAISS Deserialization</a>
     *
     * @throws IOException
     */
    public void load(IndexInput input, long totalNumberOfVectors) throws IOException {
        // Total number of vectors
        this.totalNumberOfVectors = totalNumberOfVectors;

        // We don't use `double[] assignProbas` for search. It is for index construction.
        long size = input.readLong();
        input.skipBytes(Double.BYTES * size);

        // Accumulate number of neighbor per each level.
        size = input.readLong();
        cumNumberNeighborPerLevel = new int[Math.toIntExact(size)];
        if (size > 0) {
            input.readInts(cumNumberNeighborPerLevel, 0, (int) size);
        }

        // Maximum levels per each vector
        levels = new FaissSection(input, Integer.BYTES);

        // Load `offsets` into memory.
        size = input.readLong();
        offsets = new long[(int) size];
        input.readLongs(offsets, 0, offsets.length);

        // Mark neighbor list section.
        neighbors = new FaissSection(input, Integer.BYTES);

        // HNSW graph parameters
        entryPoint = input.readInt();

        maxLevel = input.readInt();

        // Gets efConstruction. We don't use this field. It's for index building.
        input.readInt();

        efSearch = input.readInt();

        // dummy read a deprecated field.
        input.readInt();
    }
}
