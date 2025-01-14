/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss.hnsw;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.SparseFixedBitSet;
import org.opensearch.knn.partialloading.search.DistanceMaxHeap;
import org.opensearch.knn.partialloading.search.DocIdAndDistance;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;
import org.opensearch.knn.partialloading.search.ResultsCollector;
import org.opensearch.knn.partialloading.storage.Storage;
import org.opensearch.knn.partialloading.util.DistanceComputer;

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
        ResultsCollector resultsCollector,
        PartialLoadingSearchParameters searchParameters
    ) throws IOException {
        // Override `efSearch` with search parameters otherwise use default.
        final int effectiveEfSearch;
        if (searchParameters.getEfSearch() != null) {
            effectiveEfSearch = searchParameters.getEfSearch();
        } else {
            effectiveEfSearch = efSearch;
        }

        // Greedy search for the starting point at the bottom layer
        final DocIdAndDistance nearest = new DocIdAndDistance(entryPoint, distanceComputer.compute(entryPoint));
        for (int level = maxLevel; level >= 1; --level) {
            greedyUpdateNearest(searchParameters.getIndexInput(), distanceComputer, level, nearest);
        }

        // Start exhaustive search at the bottom layer.
        final SparseFixedBitSet visited = new SparseFixedBitSet(Math.toIntExact(totalNumberOfVectors));
        final int ef = Math.max(effectiveEfSearch, searchParameters.getK());
        final DistanceMaxHeap candidates = new DistanceMaxHeap(ef);
        // Put starting point in both candidates max heap and results collector.
        candidates.insertWithOverflow(nearest.id, nearest.distance);
        resultsCollector.addResult(nearest.id, nearest.distance);
        // We've visited starting point.
        visited.set(candidates.top().id);
        searchFromCandidates(searchParameters.getIndexInput(), distanceComputer, resultsCollector, candidates, visited);
    }

    private void searchFromCandidates(
        IndexInput indexInput,
        DistanceComputer distanceComputer,
        ResultsCollector resultsCollector,
        DistanceMaxHeap candidates,
        SparseFixedBitSet visited
    ) throws IOException {
        DocIdAndDistance minIad = new DocIdAndDistance(0, 0);
        while (!candidates.isEmpty()) {
            candidates.popMin(minIad);
            final long o = offsets[minIad.id];
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
                // System.out.println("neighborId=" + neighborId + ", dist=" + dist);
                resultsCollector.addResult(neighborId, dist);
                candidates.insertWithOverflow(neighborId, dist);
            }
        }
    }

    private void greedyUpdateNearest(IndexInput indexInput, DistanceComputer distanceComputer, int level, DocIdAndDistance nearest)
        throws IOException {
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
