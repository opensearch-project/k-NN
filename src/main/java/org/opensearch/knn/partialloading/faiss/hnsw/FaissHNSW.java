/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss.hnsw;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.SparseFixedBitSet;
import org.opensearch.knn.partialloading.KdyPerfCheck;
import org.opensearch.knn.partialloading.search.AbstractDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.IdAndDistance;
import org.opensearch.knn.partialloading.search.MatchDocSelector;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.search.PlainDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.distance.DistanceComputer;
import org.opensearch.knn.partialloading.storage.Storage;

import java.io.IOException;

/**
 * Ported implementation of the FAISS HNSW graph search algorithm.
 * While it follows the same steps as the original FAISS implementation, differences in how the JVM and C++ handle floating-point
 * calculations can lead to slight variations in results. However, such cases are very rare, and in most instances, the results are
 * identical to FAISS. Even when there are ranking differences, they do not impact the precision or recall of the search.
 * For more details, refer to the [FAISS HNSW implementation](
 * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.h">...</a>).
 */
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
     * Performs HNSW search logic to update results with the nearest neighbors to the query vector.
     * The process begins with a greedy search in each layer (except the bottom layer) to identify a starting point in the bottom layer.
     * Using this starting point, it performs a breadth-first search (BFS) with a max-heap based on distance to refine and narrow down
     * the adjacent neighbor vectors.
     *
     * @param distanceComputer A distance computer that accepts a vector and returns the distance between it and the query vector.
     * @param resultMaxHeap A max-heap used to collect the top-k nearest vectors based on distance.
     * @param searchParameters HNSW search parameters, including efSearch, allow customization. If efSearch is provided, it will override
     *                        the default value.
     * @param results A result array containing non-null pairs of vector IDs and their distances. After the search, it is updated by
     *                extracting elements from the result max-heap.
     * @throws IOException
     */
    public void search(
        DistanceComputer distanceComputer,
        AbstractDistanceMaxHeap resultMaxHeap,
        PartialLoadingSearchParameters searchParameters,
        IdAndDistance[] results
    ) throws IOException {
        KdyPerfCheck.setHnswEntryPoint(entryPoint);
        KdyPerfCheck.faissHnswStart();

        if (entryPoint == -1) {
            return;
        }

        IndexInput neighborListInput =
            searchParameters.getIndexInput().slice("FaissHNSWNeighborList", neighbors.getBaseOffset(), neighbors.getSectionSize());

        // Override `efSearch` with search parameters otherwise use default.
        final int effectiveEfSearch;
        if (searchParameters.getEfSearch() != null) {
            effectiveEfSearch = searchParameters.getEfSearch();
        } else {
            effectiveEfSearch = efSearch;
        }

        // Greedy search for the starting point at the bottom layer
        final IdAndDistance nearest = new IdAndDistance(entryPoint, distanceComputer.compute(entryPoint));
        final IntArray neighborList = new IntArray(16);
        for (int level = maxLevel; level >= 1; --level) {
            greedyUpdateNearest(neighborListInput, distanceComputer, level, nearest, neighborList);
        }
        KdyPerfCheck.hnswGreedyDone();
        KdyPerfCheck.setEntryPointAfterGreedy(nearest.id, nearest.distance);

        // Start exhaustive search at the bottom layer.
        final SparseFixedBitSet visited = new SparseFixedBitSet(Math.toIntExact(totalNumberOfVectors));
        final int ef = Math.max(effectiveEfSearch, searchParameters.getK());
        final PlainDistanceMaxHeap candidates = new PlainDistanceMaxHeap(ef);
        searchFromCandidates(
            neighborListInput,
            distanceComputer,
            resultMaxHeap,
            candidates,
            visited,
            searchParameters.getMatchDocSelector(),
            nearest,
            neighborList
        );
        KdyPerfCheck.hnswBfsDone();

        // Make sorted results by distance
        resultMaxHeap.orderResults(results);

        KdyPerfCheck.faissHnswDone();
    }

    /**
     * Performs a breadth-first search (BFS) using a distance-based max-heap to narrow down vectors that are approximately closest to the
     * query vector.
     * This is a ported version of the FAISS implementation, based on the paper *"Efficient and Robust Approximate Nearest Neighbor
     * Search Using Hierarchical Navigable Small World Graphs"* by Yu. A. Malkov and D. A. Yashunin.
     * For more details, refer to the [FAISS HNSW implementation](
     * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW">...</a>
     * .cpp#L584).
     *
     * @param indexInput       An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param distanceComputer A distance computer that accepts a vector and returns the distance between it and the query vector.
     * @param resultMaxHeap    A max-heap used to collect the top-k nearest vectors based on distance.
     * @param candidates       A max-distance heap for candidate vectors, where newly discovered competitive vectors are added.
     * @param visited          A bit set used to mark visited vectors during the search, ensuring each vector is visited only once.
     * @param matchDocSelector The selector determines whether a vector should be added to the results heap.
     * @param nearest          {@link IdAndDistance} instance used to track the closest vector found so far.
     * @param neighborList     Neighbor list having an integer array.
     * @throws IOException
     */
    private void searchFromCandidates(
        IndexInput indexInput,
        DistanceComputer distanceComputer,
        AbstractDistanceMaxHeap resultMaxHeap,
        PlainDistanceMaxHeap candidates,
        SparseFixedBitSet visited,
        MatchDocSelector matchDocSelector,
        IdAndDistance nearest,
        IntArray neighborList
    ) throws IOException {
        // We've visited starting point.
        visited.set(nearest.id);
        candidates.insertWithOverflow(nearest.id, nearest.distance);

        // Collect starting point in results collector.
        final boolean hasSelector = matchDocSelector != null;
        if (!hasSelector || matchDocSelector.test(nearest.id)) {
            resultMaxHeap.insertWithOverflow(nearest.id, nearest.distance);
        }

        // Start BFS searching.
        while (!candidates.isEmpty()) {
            candidates.popMin(nearest);

            final long neighborListOffset = offsets[nearest.id];
            final long neighborListStartOffset = neighborListOffset + cumNumberNeighborPerLevel[0];
            final long neighborListEndOffset = neighborListOffset + cumNumberNeighborPerLevel[1];
            // System.out.println("mid.id=" + minIad.id + ", dist=" + minIad.distance);

            neighborList.reset();
            indexInput.seek(neighborListEndOffset * Integer.BYTES);
            for (long left = neighborListEndOffset - neighborListStartOffset; left > 0 ; --left) {
                final int neighborId = indexInput.readInt();
                if (neighborId >= 0) {
                    if (visited.getAndSet(neighborId)) {
                        continue;
                    }
                    neighborList.add(neighborId);
                } else {
                    break;
                }
            }

            for (int i = 0 ; i < neighborList.i ; ++i) {
                final int neighborId = neighborList.array[i];
                final float dist = distanceComputer.compute(neighborId);
                candidates.insertWithOverflow(neighborId, dist);
                // System.out.println("neighborId=" + neighborId + ", dist=" + dist);
                if (!hasSelector || matchDocSelector.test(neighborId)) {
                    resultMaxHeap.insertWithOverflow(neighborId, dist);
                }
            }
        }
    }

    private static class IntArray {
        int[] array;
        int i;

        public IntArray(int size) {
            array = new int[size];
        }

        void reset() {
            i = 0;
        }

        void add(int value) {
            if (i < array.length) {
                array[i++] = value;
            } else {
                int[] newArray = new int[array.length * 2];
                System.arraycopy(array, 0, newArray, 0, array.length);
                array = newArray;
                array[i++] = value;
            }
        }
    }

    /**
     * Performs a greedy search at each layer to find the closest vector to the query vector. While this step focuses on a single layer,
     * the process generally continues until reaching the bottom layer. The goal of the algorithm is to quickly identify the entry point
     * in the bottom graph layer, where the actual candidate narrowing down occurs. As the name suggests, it follows the closest vector
     * found to the query vector and restarts the search with that vector in the next layer.
     *
     * @param indexInput       An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param distanceComputer A distance computer that accepts a vector and returns the distance between it and the query vector.
     * @param level            The layer level at which the greedy search will be performed.
     * @param nearest          {@link IdAndDistance} instance used to track the closed vector found so far. It will be used as an input for the
     *                         next iteration.
     * @param neighborList     Neighbor list having an integer array.
     * @throws IOException
     */
    private void greedyUpdateNearest(
        IndexInput indexInput,
        DistanceComputer distanceComputer,
        int level,
        IdAndDistance nearest,
        IntArray neighborList
    ) throws IOException {
        while (true) {
            final int prevNearest = nearest.id;

            // Determine maximum neighbor range
            final long neighborListOffset = offsets[nearest.id];
            final long neighborListStartOffset = neighborListOffset + cumNumberNeighborPerLevel[level];
            final long neighborListEndOffset = neighborListOffset + cumNumberNeighborPerLevel[level + 1];

            // System.out.println(" +++++++++++++++++++++++++ greedyUpdateNearest, begin="
            // + begin + ", end=" + end + ", prevNearest=" + prevNearest);

            neighborList.reset();
            indexInput.seek(neighborListStartOffset * Integer.BYTES);
            for (long left = neighborListEndOffset - neighborListStartOffset ; left > 0 ; --left) {
                final int neighborId = indexInput.readInt();
                if (neighborId >= 0) {
                    neighborList.add(neighborId);
                } else {
                    break;
                }
            }

            for (int i = 0 ; i < neighborList.i ; ++i) {
                final int neighborId = neighborList.array[i];
                final float distance = distanceComputer.compute(neighborId);
                if (distance < nearest.distance) {
                    nearest.id = neighborId;
                    nearest.distance = distance;
                }
            }

            // We reached a dead end. Greedy search converged, abort the loop.
            if (nearest.id == prevNearest) {
                return;
            }
        }
    }

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
    public static FaissHNSW partiallyLoad(IndexInput input, long totalNumberOfVectors) throws IOException {
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
