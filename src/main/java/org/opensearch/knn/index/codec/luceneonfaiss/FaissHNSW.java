/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Ported implementation of the FAISS HNSW graph search algorithm.
 * While it follows the same steps as the original FAISS implementation, differences in how the JVM and C++ handle floating-point
 * calculations can lead to slight variations in results. However, such cases are very rare, and in most instances, the results are
 * identical to FAISS. Even when there are ranking differences, they do not impact the precision or recall of the search.
 * For more details, refer to the [FAISS HNSW implementation](
 * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.h">...</a>).
 */
@Getter
public class FaissHNSW {
    // Cumulative number of neighbors per each level.
    private int[] cumNumberNeighborPerLevel;
    // Offset to be added to cumNumberNeighborPerLevel[level] to get the actual start offset of neighbor list.
    private long[] offsets = null;
    // Neighbor list storage.
    private final Storage neighbors = new Storage();
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
     * @return {@link FaissHNSW}, a graph search structure that represents the FAISS HNSW graph
     * @throws IOException
     */
    public static FaissHNSW load(IndexInput input, long totalNumberOfVectors) throws IOException {
        // Total number of vectors
        FaissHNSW faissHNSW = new FaissHNSW();
        faissHNSW.totalNumberOfVectors = totalNumberOfVectors;

        // We don't use `double[] assignProbas` for search. It is for index construction.
        long size = input.readLong();
        input.skipBytes(Double.BYTES * size);

        // Accumulate number of neighbor per each level.
        size = input.readLong();
        faissHNSW.cumNumberNeighborPerLevel = new int[(int) size];
        if (size > 0) {
            input.readInts(faissHNSW.cumNumberNeighborPerLevel, 0, (int) size);
        }

        // We don't use `level`.
        final Storage levels = new Storage();
        levels.markSection(input, Integer.BYTES);

        // Load `offsets` into memory.
        size = input.readLong();
        faissHNSW.offsets = new long[(int) size];
        input.readLongs(faissHNSW.offsets, 0, faissHNSW.offsets.length);

        // Mark neighbor list section.
        faissHNSW.neighbors.markSection(input, Integer.BYTES);

        // HNSW graph parameters
        faissHNSW.entryPoint = input.readInt();

        faissHNSW.maxLevel = input.readInt();

        // We don't use this field. It's for index building.
        final int efConstruction = input.readInt();

        faissHNSW.efSearch = input.readInt();

        // dummy read a deprecated field.
        input.readInt();

        return faissHNSW;
    }
}
