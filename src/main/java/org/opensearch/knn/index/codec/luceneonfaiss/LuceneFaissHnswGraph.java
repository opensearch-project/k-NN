/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class LuceneFaissHnswGraph extends HnswGraph {
    private final FaissHNSW faissHnsw;
    private final IndexInput indexInput;
    private final int numVectors;
    private int[] neighborIdList;
    private int numNeighbors;
    private int nextNeighborIndex;

    public LuceneFaissHnswGraph(FaissHNSWFlatIndex hnswFlatIndex, IndexInput indexInput) {
        this.faissHnsw = hnswFlatIndex.getHnsw();
        this.indexInput = indexInput;
        this.numVectors = (int) hnswFlatIndex.getStorage().getTotalNumberOfVectors();
    }

    @Override
    public void seek(int level, int target) {
        // Get a relative starting offset of neighbor list at `level`.
        long o = faissHnsw.getOffsets()[target];

        // `begin` and `end` represent for a pair of staring offset and end offset.
        // But, what `end` represents is the maximum offset a neighbor list at a level can have.
        // Therefore, it is required to traverse a list until getting a terminal `-1`.
        final long begin = o + faissHnsw.getCumNumberNeighborPerLevel()[level];
        final long end = o + faissHnsw.getCumNumberNeighborPerLevel()[level + 1];
        loadNeighborIdList(begin, end);
    }

    private void loadNeighborIdList(final long begin, final long end) {
        // Make sure we have sufficient space for neighbor list
        final long maxLength = end - begin;
        if (neighborIdList == null || neighborIdList.length < maxLength) {
            neighborIdList = new int[(int) (maxLength * 1.5)];
        }

        // Seek to the first offset of neighbor list
        try {
            indexInput.seek(faissHnsw.getNeighbors().getBaseOffset() + Integer.BYTES * begin);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Fill the array with neighbor ids
        int index = 0;
        try {
            for (long i = begin; i < end; i++) {
                final int neighborId = indexInput.readInt();
                if (neighborId >= 0) {
                    neighborIdList[index++] = neighborId;
                } else {
                    break;
                }
            }

            // Set variables for navigation
            numNeighbors = index;
            nextNeighborIndex = 0;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int size() {
        return numVectors;
    }

    @Override
    public int nextNeighbor() {
        if (nextNeighborIndex < numNeighbors) {
            return neighborIdList[nextNeighborIndex++];
        }

        // Neighbor list has been exhausted.
        return NO_MORE_DOCS;
    }

    @Override
    public int numLevels() {
        return faissHnsw.getMaxLevel();
    }

    @Override
    public int entryNode() {
        return faissHnsw.getEntryPoint();
    }

    @Override
    public NodesIterator getNodesOnLevel(int i) {
        throw new UnsupportedOperationException();
    }
}
