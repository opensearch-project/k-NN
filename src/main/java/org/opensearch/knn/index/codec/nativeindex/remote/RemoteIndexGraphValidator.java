/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWProvider;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHnswGraph;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;
import java.util.Arrays;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/** Validates connectivity before a remotely built graph is accepted. */
final class RemoteIndexGraphValidator {
    private RemoteIndexGraphValidator() {}

    /** A detected graph fault, distinct from an I/O failure while reading the staged artifact. */
    static final class GraphFaultException extends IOException {
        GraphFaultException(String message) {
            super(message);
        }
    }

    static void validate(IndexInput input) throws IOException {
        FaissIndex index = FaissIndex.load(input);
        if (index instanceof FaissHNSWProvider == false) {
            throw new GraphFaultException("Remote index does not contain an HNSW graph");
        }
        FaissHnswGraph graph = new FaissHnswGraph(((FaissHNSWProvider) index).getFaissHnsw(), input);
        // Scan level 0 in file order, treating an edge in either direction as a connection.
        // Union by size with path compression needs only one int per vector and avoids loading edges or vectors into memory.
        int[] parents = new int[graph.size()];
        Arrays.fill(parents, -1);
        int components = graph.size();
        for (int node = 0; node < graph.size(); node++) {
            graph.seek(0, node);
            for (int neighbor = graph.nextNeighbor(); neighbor != NO_MORE_DOCS; neighbor = graph.nextNeighbor()) {
                if (neighbor >= graph.size()) {
                    throw new GraphFaultException("Remote graph contains an invalid neighbor ID: " + neighbor);
                }
                int nodeRoot = root(parents, node);
                int neighborRoot = root(parents, neighbor);
                if (nodeRoot != neighborRoot) {
                    if (parents[nodeRoot] > parents[neighborRoot]) {
                        int temp = nodeRoot;
                        nodeRoot = neighborRoot;
                        neighborRoot = temp;
                    }
                    parents[nodeRoot] += parents[neighborRoot];
                    parents[neighborRoot] = nodeRoot;
                    components--;
                }
            }
        }
        if (components > 1) {
            throw new GraphFaultException("Remote graph has " + components + " connected components; falling back to CPU indexing");
        }
    }

    private static int root(int[] parents, int node) {
        while (parents[node] >= 0) {
            int parent = parents[node];
            if (parents[parent] >= 0) {
                parents[node] = parents[parent];
            }
            node = parent;
        }
        return node;
    }
}
