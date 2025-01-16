/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss.hnsw;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.SparseFixedBitSet;
import org.opensearch.knn.partialloading.search.AbstractDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.DocIdAndDistance;
import org.opensearch.knn.partialloading.search.MatchDocSelector;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.search.PlainDistanceMaxHeap;
import org.opensearch.knn.partialloading.search.distance.DistanceComputer;
import org.opensearch.knn.partialloading.storage.Storage;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class FaissHNSW {
    private int[] cumNumberNeighborPerLevel;
    private long[] offsets = null;
    private final Storage neighbors = new Storage();
    private int entryPoint;
    private int maxLevel = -1;
    private int efSearch = 16;
    private long totalNumberOfVectors;

    public void search(
        DistanceComputer distanceComputer,
        AbstractDistanceMaxHeap resultMaxHeap,
        PartialLoadingSearchParameters searchParameters,
        DocIdAndDistance[] results
    ) throws IOException {
        // Override `efSearch` with search parameters otherwise use default.
        final int effectiveEfSearch;
        if (searchParameters.getEfSearch() != null) {
            effectiveEfSearch = searchParameters.getEfSearch();
        } else {
            effectiveEfSearch = efSearch;
        }

        // Greedy search for the starting point at the bottom layer
        final SparseFixedBitSet visited = new SparseFixedBitSet(Math.toIntExact(totalNumberOfVectors));
        final DocIdAndDistance nearest = new DocIdAndDistance(entryPoint, distanceComputer.compute(entryPoint));
        for (int level = maxLevel; level >= 1; --level) {
            greedyUpdateNearest(searchParameters.getIndexInput(), distanceComputer, visited, level, nearest);
        }

        // Start exhaustive search at the bottom layer.
        visited.clear();
        final int ef = Math.max(effectiveEfSearch, searchParameters.getK());
        final PlainDistanceMaxHeap candidates = new PlainDistanceMaxHeap(ef);
        searchFromCandidates(
            searchParameters.getIndexInput(),
            distanceComputer,
            resultMaxHeap,
            candidates,
            visited,
            searchParameters.getMatchDocSelector(),
            nearest
        );

        // Make sorted results by distance
        resultMaxHeap.orderResults(results);
    }

    private void searchFromCandidates(
        IndexInput indexInput,
        DistanceComputer distanceComputer,
        AbstractDistanceMaxHeap resultMaxHeap,
        PlainDistanceMaxHeap candidates,
        SparseFixedBitSet visited,
        MatchDocSelector matchDocSelector,
        DocIdAndDistance nearest
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
            final long o = offsets[nearest.id];
            final long begin = o + cumNumberNeighborPerLevel[0];
            final long end = o + cumNumberNeighborPerLevel[1];
            // System.out.println("mid.id=" + minIad.id + ", dist=" + minIad.distance);

            for (long offset = begin; offset < end; offset++) {
                final int neighborId = neighbors.readInt(indexInput, offset);
                if (neighborId < 0) {
                    break;
                }
                if (visited.getAndSet(neighborId)) {
                    continue;
                }
                final float dist = distanceComputer.compute(neighborId);
                candidates.insertWithOverflow(neighborId, dist);
                // System.out.println("neighborId=" + neighborId + ", dist=" + dist);
                if (!hasSelector || matchDocSelector.test(neighborId)) {
                    resultMaxHeap.insertWithOverflow(neighborId, dist);
                }
            }
        }
    }

    private void greedyUpdateNearest(
        IndexInput indexInput,
        DistanceComputer distanceComputer,
        SparseFixedBitSet visited,
        int level,
        DocIdAndDistance nearest
    ) throws IOException {
        while (true) {
            final int prevNearest = nearest.id;

            // Determine maximum neighbor range
            final long neighborListOffset = offsets[nearest.id];
            final long neighborListStartOffset = neighborListOffset + cumNumberNeighborPerLevel[level];
            final long neighborListEndOffset = neighborListOffset + cumNumberNeighborPerLevel[level + 1];

            // System.out.println(" +++++++++++++++++++++++++ greedyUpdateNearest, begin="
            // + begin + ", end=" + end + ", prevNearest=" + prevNearest);

            for (long i = neighborListStartOffset; i < neighborListEndOffset; ++i) {
                final int neighborId = neighbors.readInt(indexInput, i);
                if (neighborId >= 0) {
                    // We don't need to visit a vector more than once.
                    // Because, a visited node is either the closed vector to query vector that we've found so far
                    // or, it's a farther vector than the best one that no need to reconsider.
                    // Since this is a greedy algorithm, it is so much enough to track the best found vector.
                    if (visited.getAndSet(neighborId)) {
                        continue;
                    }
                    final float distance = distanceComputer.compute(neighborId);
                    if (distance < nearest.distance) {
                        nearest.id = neighborId;
                        nearest.distance = distance;
                    }
                } else {
                    // Reached end of neighbor list.
                    break;
                }
            }

            if (nearest.id == prevNearest) {
                return;
            }
        }
    }

    public static FaissHNSW readHnsw(IndexInput input, long totalNumberOfVectors) throws IOException {
        FaissHNSW faissHNSW = new FaissHNSW();
        faissHNSW.totalNumberOfVectors = totalNumberOfVectors;

        long size = input.readLong();

        // We don't use `assignProbas`
        final double[] assignProbas = new double[(int) size];
        byte[] doubleBytes = new byte[8];
        for (int i = 0; i < size; i++) {
            input.readBytes(doubleBytes, 0, 8);
            assignProbas[i] = ByteBuffer.wrap(doubleBytes).order(ByteOrder.LITTLE_ENDIAN).getDouble();
        }

        size = input.readLong();
        faissHNSW.cumNumberNeighborPerLevel = new int[(int) size];
        if (size > 0) {
            input.readInts(faissHNSW.cumNumberNeighborPerLevel, 0, (int) size);
        }

        // We don't use `level`.
        final Storage levels = new Storage();
        levels.markSection(input, Integer.BYTES);

        size = input.readLong();
        faissHNSW.offsets = new long[(int) size];
        input.readLongs(faissHNSW.offsets, 0, faissHNSW.offsets.length);

        faissHNSW.neighbors.markSection(input, Integer.BYTES);

        faissHNSW.entryPoint = input.readInt();

        faissHNSW.maxLevel = input.readInt();

        // We don't use this field.
        final int efConstruction = input.readInt();

        faissHNSW.efSearch = input.readInt();

        // dummy read a deprecated field.
        input.readInt();

        return faissHNSW;
    }
}
