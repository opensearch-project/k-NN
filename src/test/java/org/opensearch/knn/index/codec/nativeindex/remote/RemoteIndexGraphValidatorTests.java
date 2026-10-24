/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWProvider;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class RemoteIndexGraphValidatorTests extends KNNTestCase {
    private static final List<String> CAGRA_FILES = List.of(
        "faiss_cagra_flat_float_300_vectors_768_dims.bin",
        "faiss_cagra2_flat_fp16_300_vectors_768_dims.bin",
        "faiss_cagra2_flat_bf16_300_vectors_768_dims.bin",
        "faiss_cagra2_flat_byte_300_vectors_768_dims.bin",
        "faiss_cagra2_flat_binary_300_vectors_768_dims.bin",
        "faiss_binary_cagra_300_vectors_32_dims.faiss"
    );

    public void testConnectedCagraGraphs() throws IOException {
        for (String file : CAGRA_FILES) {
            validate(graphBytes(file, 1));
        }
    }

    public void testDisconnectedCagraGraphs() throws IOException {
        for (String file : CAGRA_FILES) {
            IOException exception = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(graphBytes(file, 2)));
            assertTrue(exception.getMessage(), exception.getMessage().contains("2 connected components"));
        }
    }

    public void testIsolatedVertices() throws IOException {
        IOException exception = expectThrows(
            RemoteIndexGraphValidator.GraphFaultException.class,
            () -> validate(graphBytes(CAGRA_FILES.getFirst(), 300))
        );
        assertTrue(exception.getMessage(), exception.getMessage().contains("300 connected components"));
    }

    public void testSingleVertex() throws IOException {
        // This fixture is a raw HNSW section. Wrap it in a CAGRA header with skipped vector storage.
        byte[] hnsw;
        try (InputStream resource = getClass().getResourceAsStream("/data/memoryoptsearch/faiss_hnsw_one_vector.bin")) {
            hnsw = resource.readAllBytes();
        }
        ByteBuffer buffer = ByteBuffer.allocate(47 + hnsw.length).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("IHNc".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(8).putLong(1).putLong(0).putLong(0).put((byte) 1).putInt(1);
        buffer.put((byte) 0).put((byte) 1).putInt(32);
        buffer.put(hnsw).put("null".getBytes(StandardCharsets.US_ASCII));
        validate(buffer.array());
    }

    public void testInvalidNeighbor() throws IOException {
        byte[] bytes = graphBytes(CAGRA_FILES.getFirst(), 1);
        try (ByteArrayIndexInput input = new ByteArrayIndexInput("graph", bytes)) {
            FaissHNSW graph = ((FaissHNSWProvider) FaissIndex.load(input)).getFaissHnsw();
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(Math.toIntExact(graph.getNeighbors().getBaseOffset()), 300);
        }
        IOException exception = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
        assertTrue(exception.getMessage(), exception.getMessage().contains("invalid neighbor ID"));
    }

    private static void validate(byte[] bytes) throws IOException {
        try (ByteArrayIndexInput input = new ByteArrayIndexInput("graph", bytes)) {
            RemoteIndexGraphValidator.validate(input);
        }
    }

    // Keep the real serialized CAGRA formats and ID maps, replacing only level-0 edges with deterministic components.
    // Each component is a chain pointing towards its first vertex, so connectivity must honor edges in either direction.
    static byte[] graphBytes(String file, int components) throws IOException {
        byte[] bytes;
        try (InputStream resource = RemoteIndexGraphValidatorTests.class.getResourceAsStream("/data/memoryoptsearch/" + file)) {
            bytes = resource.readAllBytes();
        }
        try (ByteArrayIndexInput input = new ByteArrayIndexInput("graph", bytes)) {
            FaissHNSW graph = ((FaissHNSWProvider) FaissIndex.load(input)).getFaissHnsw();
            ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            int size = Math.toIntExact(graph.getTotalNumberOfVectors());
            for (int node = 0; node < size; node++) {
                long start = graph.getOffsetsReader().get(node) + graph.getCumNumberNeighborPerLevel()[0];
                long offset = graph.getNeighbors().getBaseOffset() + Integer.BYTES * start;
                for (int neighbor = 0; neighbor < graph.getMaxNumNeighbors(); neighbor++) {
                    int id = neighbor == 0 && node >= components ? node - components : -1;
                    buffer.putInt(Math.toIntExact(offset + Integer.BYTES * neighbor), id);
                }
            }
        }
        return bytes;
    }
}
