/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Objects;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * This graph implements Lucene's HNSW graph interface using the FAISS HNSW graph. Conceptually, both libraries represent the graph
 * similarly, maintaining a list of neighbor IDs. This implementation acts as a bridge, enabling Lucene's HNSW graph searcher to perform
 * vector searches on a FAISS index.
 * <p>
 * NOTE: This is not thread safe. It should be created every time in {@link KnnVectorsReader}.search likewise
 * <a href="https://github.com/apache/lucene/blob/92290a0201458152c9e03d199f38f2e8a479f045/lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsReader.java#L467">OffHeapHnswGraph</a>
 * in Lucene.
 */
public class FaissHnswGraph extends HnswGraph {
    private final FaissHNSW faissHnsw;
    private final IndexInput indexInput;
    private final int numVectors;
    private int[] neighborIdList;
    private int numNeighbors;
    private int nextNeighborIndex;

    public FaissHnswGraph(final FaissHNSW faissHNSW, final IndexInput indexInput) {
        this.faissHnsw = faissHNSW;
        // Offset readers MUST non null.
        Objects.requireNonNull(faissHNSW.getOffsetsReader());
        this.indexInput = indexInput;
        this.numVectors = Math.toIntExact(faissHNSW.getTotalNumberOfVectors());
    }

    /**
     * Seek to the starting offset of neighbor ids at the given `level`. In which, it will load all ids into a buffer array.
     * @param level The level of graph
     * @param internalVectorId An internal vector id.
     */
    @Override
    public void seek(int level, int internalVectorId) {
        // Get a relative starting offset of neighbor list at `level`.
        final long o = faissHnsw.getOffsetsReader().get(internalVectorId);

        // `begin` and `end` represent for a pair of staring offset and end offset.
        // But, what `end` represents is the maximum offset a neighbor list at a level can have.
        // Therefore, it is required to traverse a list until getting a terminal `-1`.
        // Ex: [1, 5, 20, 100, -1, -1, ..., -1]
        final long begin = o + faissHnsw.getCumNumberNeighborPerLevel()[level];
        final long end = o + faissHnsw.getCumNumberNeighborPerLevel()[level + 1];
        loadNeighborIdList(begin, end);
    }

    private void loadNeighborIdList(final long begin, final long end) {
        // Make sure we have sufficient space for neighbor list
        final long maxLength = end - begin;
        if (neighborIdList == null || neighborIdList.length < maxLength) {
            neighborIdList = new int[(int) (maxLength)];
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
                // The idea is that a vector does not always have a complete list of neighbor vectors.
                // FAISS assigns a fixed size to the neighbor list and uses -1 to indicate missing entries.
                // Therefore, we can safely stop once hit -1.
                // For example, if the neighbor list size is 16 and a vector has only 8 neighbors, the list would appear as:
                // [1, 4, 6, 8, 13, 17, 60, 88, -1, -1, ..., -1].
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
    public NodesIterator getNodesOnLevel(final int level) {
        try {
            // Prepare input stream to `level` section.
            final FaissSection levelsSection = faissHnsw.getLevels();
            final IndexInput levelIndexInput = indexInput.clone();
            levelIndexInput.seek(levelsSection.getBaseOffset());

            // Count the number of vectors at the level.
            int numVectorsAtLevel = 0;
            for (int i = 0; i < numVectors; ++i) {
                final int maxLevel = levelIndexInput.readInt();
                // Note that maxLevel=3 indicates that a vector exists level-0 (bottom), level-1 and level-2.
                if (maxLevel > level) {
                    ++numVectorsAtLevel;
                }
            }

            // Return iterator
            levelIndexInput.seek(levelsSection.getBaseOffset());
            return new NodesIterator(numVectorsAtLevel) {
                int vectorNo = -1;
                int numVisitedVectors = 0;

                @Override
                public boolean hasNext() {
                    return numVisitedVectors < size;
                }

                @Override
                public int nextInt() {
                    while (true) {
                        try {
                            // Advance
                            ++vectorNo;
                            final int maxLevel = levelIndexInput.readInt();

                            // Check the level
                            if (maxLevel > level) {
                                ++numVisitedVectors;
                                return vectorNo;
                            }
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    }
                }

                @Override
                public int consume(int[] ints) {
                    if (hasNext() == false) {
                        throw new NoSuchElementException();
                    }
                    final int copySize = Math.min(size - numVisitedVectors, ints.length);
                    for (int i = 0; i < copySize; ++i) {
                        ints[i] = nextInt();
                    }
                    return copySize;
                }
            };
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int neighborCount() {
        return numNeighbors;
    }

    @Override
    public int maxConn() {
        return UNKNOWN_MAX_CONN;
    }
}
