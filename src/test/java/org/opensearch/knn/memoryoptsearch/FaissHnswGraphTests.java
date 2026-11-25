/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.ArgumentCaptor;
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

import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public class FaissHnswGraphTests extends KNNTestCase {
    // HNSW graph having two levels. entry_point=12, max_level=1
    private static final String HNSW_100_VECTORS_PATH = "data/memoryoptsearch/faiss_hnsw_100_vectors.bin";
    // HNSW graph having the bottom level only. entry_point=0, max_level=0
    private static final String HNSW_ONE_VECTOR_PATH = "data/memoryoptsearch/faiss_hnsw_one_vector.bin";
    private static final int NUM_VECTORS_ONE_HUNDRED = 100;
    private static final int NUM_VECTORS_ONE = 1;

    @SneakyThrows
    public void testTraverseHnswGraph() {
        final FaissHnswGraph graph = prepareFaissHnswGraph(HNSW_100_VECTORS_PATH, NUM_VECTORS_ONE_HUNDRED);

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
        final FaissHnswGraph graph = prepareFaissHnswGraph(HNSW_100_VECTORS_PATH, NUM_VECTORS_ONE_HUNDRED);
        // Iterate all vectors at level-0
        HnswGraph.NodesIterator iterator = graph.getNodesOnLevel(0);
        Set<Integer> vectorIds = new HashSet<>();
        while (iterator.hasNext()) {
            vectorIds.add(iterator.next());
        }
        assertEquals(NUM_VECTORS_ONE_HUNDRED, vectorIds.size());
        for (int i = 0; i < NUM_VECTORS_ONE_HUNDRED; ++i) {
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

    public void testNumLevelsAndEntryNode() {
        // FAISS `max_level` is the 0-based index of the top level, so the number of levels is `max_level + 1`.
        // entry_point=12, max_level=1 -> 2 levels
        final FaissHnswGraph graph = prepareFaissHnswGraph(HNSW_100_VECTORS_PATH, NUM_VECTORS_ONE_HUNDRED);
        assertEquals(2, graph.numLevels());
        assertEquals(12, graph.entryNode());

        // The entry point must exist at the top level.
        final HnswGraph.NodesIterator topLevelNodes = graph.getNodesOnLevel(graph.numLevels() - 1);
        final Set<Integer> topLevelVectorIds = new HashSet<>();
        while (topLevelNodes.hasNext()) {
            topLevelVectorIds.add(topLevelNodes.next());
        }
        assertTrue(topLevelVectorIds.contains(graph.entryNode()));

        // entry_point=0, max_level=0 -> 1 level
        final FaissHnswGraph singleLevelGraph = prepareFaissHnswGraph(HNSW_ONE_VECTOR_PATH, NUM_VECTORS_ONE);
        assertEquals(1, singleLevelGraph.numLevels());
        assertEquals(0, singleLevelGraph.entryNode());
    }

    @SneakyThrows
    public void testLuceneSearcherStartsDescentAtTopLevelEntryPoint() {
        // Test graph has entry_point=12, max_level=1. Spy on the graph to record every seek(level, node) that the searcher makes.
        final FaissHnswGraph graph = Mockito.spy(prepareFaissHnswGraph(HNSW_100_VECTORS_PATH, NUM_VECTORS_ONE_HUNDRED));

        // Derive a made-up score from the node id; the first seek does not depend on the scores.
        // The fixture holds only the graph, not the vectors, so real distances cannot be computed.
        final RandomVectorScorer scorer = new RandomVectorScorer() {
            @Override
            public float score(final int node) {
                return 1f / (1 + node);
            }

            @Override
            public int maxOrd() {
                return NUM_VECTORS_ONE_HUNDRED;
            }
        };

        HnswGraphSearcher.search(scorer, new TopKnnCollector(10, Integer.MAX_VALUE), graph, null);

        final ArgumentCaptor<Integer> levels = ArgumentCaptor.forClass(Integer.class);
        final ArgumentCaptor<Integer> nodes = ArgumentCaptor.forClass(Integer.class);
        verify(graph, atLeastOnce()).seek(levels.capture(), nodes.capture());

        // The descent must start at the FAISS top level (max_level=1), on the entry point (12).
        assertEquals(1, levels.getAllValues().get(0).intValue());
        assertEquals(12, nodes.getAllValues().get(0).intValue());
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
    private static FaissHnswGraph prepareFaissHnswGraph(final String relativePath, final int totalNumberOfVectors) {
        // Prepare parent index
        final FaissHNSWIndex parentIndex = Mockito.mock(FaissHNSWIndex.class);
        IndexInput indexInput = loadHnswBinary(relativePath);

        // Prepare FaissHNSW
        final FaissHNSW faissHNSW = new FaissHNSW();
        faissHNSW.load(indexInput, totalNumberOfVectors);
        when(parentIndex.getFaissHnsw()).thenReturn(faissHNSW);
        when(parentIndex.getTotalNumberOfVectors()).thenReturn(totalNumberOfVectors);

        // Create LuceneFaissHnswGraph
        indexInput = loadHnswBinary(relativePath);
        final FaissHnswGraph graph = new FaissHnswGraph(faissHNSW, indexInput);
        return graph;
    }

    private static final int[] FIRST_NEIGHBOR_LIST_AT_0_LEVEL = new int[] { 25, 10, 11, 16, 82 };
    private static final int[] NINETY_NINETH_NEIGHBOR_LIST_AT_0_LEVEL = new int[] { 79, 14, 51, 42, 87, 11, 34, 60, 77, 46, 37, 62 };
    private static final int[] FIRST_NEIGHBOR_LIST_AT_1_LEVEL = new int[] { 51, 31, 10, 33, 11, 23, 97, 16, 65, 32, 24, 98 };
}
