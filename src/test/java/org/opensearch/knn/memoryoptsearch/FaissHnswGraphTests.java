/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHnswGraph;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public class FaissHnswGraphTests extends KNNTestCase {
    private static final int NUM_VECTORS = 100;

    @SneakyThrows
    public void testTraverseHnswGraph() {
        final FaissHnswGraph graph = prepareFaissHnswGraph();

        // Validate graph
        graph.seek(0, 0);
        assertArrayEquals(FIRST_NEIGHBOR_LIST_AT_0_LEVEL, getNeighborIdList(graph));

        graph.seek(0, 99);
        assertArrayEquals(NINETY_NINETH_NEIGHBOR_LIST_AT_0_LEVEL, getNeighborIdList(graph));

        graph.seek(1, 0);
        assertArrayEquals(FIRST_NEIGHBOR_LIST_AT_1_LEVEL, getNeighborIdList(graph));
    }

    @SneakyThrows
    public void testNodesIterator() {
        final FaissHnswGraph graph = prepareFaissHnswGraph();
        // Iterate all vectors at level-0
        HnswGraph.NodesIterator iterator = graph.getNodesOnLevel(0);
        Set<Integer> vectorIds = new HashSet<>();
        while (iterator.hasNext()) {
            vectorIds.add(iterator.next());
        }
        assertEquals(NUM_VECTORS, vectorIds.size());
        for (int i = 0; i < NUM_VECTORS; ++i) {
            assertTrue(vectorIds.contains(i));
        }

        // Test bulk
        int[] buffer = new int[37];
        iterator = graph.getNodesOnLevel(0);

        // Copied 37/100
        int copied = iterator.consume(buffer);
        assertEquals(buffer.length, copied);

        // Copied 74/100
        copied = iterator.consume(buffer);
        assertEquals(buffer.length, copied);

        // Copied 26 more, 100/100.
        copied = iterator.consume(buffer);
        assertEquals(26, copied);

        try {
            iterator.consume(buffer);
            fail();
        } catch (NoSuchElementException e) {
            // exhausted
        }
    }

    @SneakyThrows
    private static int[] getNeighborIdList(final FaissHnswGraph graph) {
        final List<Integer> neighborIds = new ArrayList<>();
        while (true) {
            final int vectorId = graph.nextNeighbor();
            if (vectorId != DocIdSetIterator.NO_MORE_DOCS) {
                neighborIds.add(vectorId);
            } else {
                break;
            }
        }

        return neighborIds.stream().mapToInt(i -> i).toArray();
    }

    @SneakyThrows
    private static FaissHnswGraph prepareFaissHnswGraph() {
        // Prepare parent index
        final FaissHNSWIndex parentIndex = Mockito.mock(FaissHNSWIndex.class);
        IndexInput indexInput = loadHnswBinary("data/memoryoptsearch/faiss_hnsw_100_vectors.bin");

        // Prepare FaissHNSW
        final int totalNumberOfVectors = 100;
        final FaissHNSW faissHNSW = new FaissHNSW();
        faissHNSW.load(indexInput, totalNumberOfVectors);
        when(parentIndex.getHnsw()).thenReturn(faissHNSW);
        when(parentIndex.getTotalNumberOfVectors()).thenReturn(totalNumberOfVectors);

        // Create LuceneFaissHnswGraph
        indexInput = loadHnswBinary("data/memoryoptsearch/faiss_hnsw_100_vectors.bin");
        final FaissHnswGraph graph = new FaissHnswGraph(faissHNSW, indexInput);
        return graph;
    }

    private static final int[] FIRST_NEIGHBOR_LIST_AT_0_LEVEL = new int[] { 25, 10, 11, 16, 82 };
    private static final int[] NINETY_NINETH_NEIGHBOR_LIST_AT_0_LEVEL = new int[] { 79, 14, 51, 42, 87, 11, 34, 60, 77, 46, 37, 62 };
    private static final int[] FIRST_NEIGHBOR_LIST_AT_1_LEVEL = new int[] { 51, 31, 10, 33, 11, 23, 97, 16, 65, 32, 24, 98 };
}
