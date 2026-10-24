/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.index.MergeAbortChecker;
import org.mockito.MockedStatic;
import org.opensearch.knn.index.codec.nativeindex.IndexBuildAbortedException;
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
import java.util.Arrays;

import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.mockStatic;

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

    public void testUnmodifiedFaissFixtures() throws IOException {
        for (String file : CAGRA_FILES) {
            try (InputStream resource = getClass().getResourceAsStream("/data/memoryoptsearch/" + file)) {
                validate(resource.readAllBytes());
            }
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
        validate(buffer.array(), 1);
    }

    public void testInvalidNeighbor() throws IOException {
        byte[] bytes = graphBytesWithInvalidNeighbor(CAGRA_FILES.getFirst(), 300);
        IOException exception = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
        assertTrue(exception.getMessage(), exception.getMessage().contains("invalid neighbor ID"));
    }

    public void testInvalidNeighborMatchingLuceneSentinel() throws IOException {
        for (String file : CAGRA_FILES) {
            byte[] bytes = graphBytesWithInvalidNeighbor(file, Integer.MAX_VALUE);
            IOException exception = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
            assertTrue(exception.getMessage(), exception.getMessage().contains("invalid neighbor ID"));
        }
    }

    public void testInvalidGraphVectorCounts() throws IOException {
        for (String file : CAGRA_FILES) {
            for (long count : new long[] { -1, 0, 299, Integer.MAX_VALUE, Long.MAX_VALUE }) {
                byte[] bytes = graphBytesWithVectorCount(file, count);
                expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
            }
        }
    }

    public void testIdMapVectorCountMustMatchBuildRequest() throws IOException {
        byte[] bytes = graphBytes(CAGRA_FILES.getFirst(), 1);
        assertEquals("IxMp", new String(bytes, 0, 4, StandardCharsets.US_ASCII));
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(8, 0);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
    }

    public void testVectorCountMustMatchSerializedLevels() throws IOException {
        byte[] bytes = graphBytesWithVectorCount(CAGRA_FILES.getFirst(), 299);
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(8, 299);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes, 299));
    }

    public void testTruncatedGraphIsGraphFault() {
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate("IHNc".getBytes(StandardCharsets.US_ASCII)));
    }

    public void testUnsupportedIndexTypeIsGraphFault() {
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate("bad!".getBytes(StandardCharsets.US_ASCII)));
    }

    public void testParserIoFailureIsNotGraphFault() throws IOException {
        IOException failure = new IOException("Disk read failed");
        try (IndexInput input = spy(new ByteArrayIndexInput("graph", graphBytes(CAGRA_FILES.getFirst(), 1)))) {
            doThrow(failure).when(input).readInt();
            assertSame(failure, expectThrows(IOException.class, () -> RemoteIndexGraphValidator.validate(input, 300)));
        }
    }

    public void testGraphIoFailureIsNotGraphFault() throws IOException {
        byte[] bytes = graphBytes(CAGRA_FILES.getFirst(), 1);
        long neighborsOffset;
        try (ByteArrayIndexInput input = new ByteArrayIndexInput("graph", bytes)) {
            FaissHNSW graph = ((FaissHNSWProvider) FaissIndex.load(input)).getFaissHnsw();
            neighborsOffset = graph.getNeighbors().getBaseOffset();
        }
        IOException failure = new IOException("Disk seek failed");
        try (IndexInput input = spy(new ByteArrayIndexInput("graph", bytes))) {
            doThrow(failure).when(input).seek(neighborsOffset);
            assertSame(failure, expectThrows(IOException.class, () -> RemoteIndexGraphValidator.validate(input, 300)));
        }
    }

    public void testHandWrittenCagraGraph() throws IOException {
        validate(smallGraph(), 3);
    }

    public void testLargeSerializedGraph() throws IOException {
        validate(ringGraph(100_000), 100_000);
    }

    public void testOversizedCumulativeTableIsRejectedBeforeAllocation() {
        byte[] bytes = smallGraph();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(59, Integer.MAX_VALUE);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes, 3));
    }

    public void testOversizedDegreeIsRejectedBeforeAllocation() {
        byte[] bytes = smallGraph();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(71, Integer.MAX_VALUE);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes, 3));
    }

    public void testTerminalOffsetMustMatchNeighborSection() {
        byte[] bytes = smallGraph();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(127, 7);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes, 3));
    }

    public void testIdMapMustContainAllVectorIds() {
        byte[] graph = smallGraph();
        ByteBuffer bytes = ByteBuffer.allocate(37 + graph.length + 8).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IxMp".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(1).putLong(3).putLong(0).putLong(0).put((byte) 1).putInt(1);
        bytes.put(graph).putLong(0);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes.array(), 3));
    }

    public void testNestedGraphCannotBeUsedAsFlatStorage() {
        byte[] graph = smallGraph();
        ByteBuffer bytes = ByteBuffer.allocate(graph.length * 2 - 4);
        bytes.put(graph, 0, graph.length - 4).put(graph);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes.array(), 3));
    }

    public void testValidationHonorsMergeCancellation() throws Exception {
        try (MockedStatic<MergeAbortChecker> mergeAbort = mockStatic(MergeAbortChecker.class)) {
            mergeAbort.when(MergeAbortChecker::isMergeAborted).thenReturn(true);
            expectThrows(IndexBuildAbortedException.class, () -> validate(smallGraph(), 3));
        }
    }

    public void testValidationHonorsCancellationDuringScan() throws Exception {
        try (MockedStatic<MergeAbortChecker> mergeAbort = mockStatic(MergeAbortChecker.class)) {
            mergeAbort.when(MergeAbortChecker::isMergeAborted).thenReturn(false, false, false, true);
            expectThrows(IndexBuildAbortedException.class, () -> validate(ringGraph(2048), 2048));
        }
    }

    public void testEveryTruncatedPrefixIsRejected() {
        byte[] graph = smallGraph();
        for (int length = 0; length < graph.length; length++) {
            byte[] prefix = Arrays.copyOf(graph, length);
            expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(prefix, 3));
        }
    }

    public void testSectionLengthOverflowIsRejected() {
        for (int offset : new int[] { 43, 59, 75, 95, 135 }) {
            for (long count : new long[] { -1, Long.MIN_VALUE, Long.MAX_VALUE, 1L << 62 }) {
                byte[] bytes = smallGraph();
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(offset, count);
                expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes, 3));
            }
        }
    }

    public void testUpperLevelNeighborMustExistOnThatLevel() {
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(smallGraphWithUpperLevel(1), 3));
    }

    public void testValidUpperLevelGraph() throws IOException {
        validate(smallGraphWithUpperLevel(0), 3);
    }

    public void testHnswEntryPointMustExistAtMaximumLevel() {
        byte[] cagra = smallGraphWithUpperLevel(0);
        // Plain HNSW has the same graph sections without the six CAGRA-specific header bytes.
        ByteBuffer bytes = ByteBuffer.allocate(cagra.length - 6).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IHNf".getBytes(StandardCharsets.US_ASCII));
        bytes.put(cagra, 4, 33).put(cagra, 43, cagra.length - 43);
        bytes.putInt(169, 1); // Node 1 only belongs to level 0; maximum level is 1.
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes.array(), 3));
    }

    private static byte[] smallGraphWithUpperLevel(int upperNeighbor) {
        ByteBuffer bytes = ByteBuffer.allocate(199).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IHNc".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(1).putLong(3).putLong(0).putLong(0).put((byte) 1).putInt(1);
        bytes.put((byte) 0).put((byte) 1).putInt(32);
        bytes.putLong(1).putDouble(1.0);
        bytes.putLong(3).putInt(0).putInt(2).putInt(3);
        bytes.putLong(3).putInt(2).putInt(1).putInt(1);
        bytes.putLong(4).putLong(0).putLong(3).putLong(5).putLong(7);
        bytes.putLong(7).putInt(1).putInt(-1).putInt(upperNeighbor).putInt(2).putInt(-1).putInt(0).putInt(-1);
        bytes.putInt(0).putInt(1).putInt(16).putInt(16).putInt(0);
        bytes.put("null".getBytes(StandardCharsets.US_ASCII));
        return bytes.array();
    }

    // Independent serialized CAGRA fixture: 3 nodes in a directed ring, degree 2, skipped flat storage.
    // The locations used by the malformed-input tests follow this hand-written layout, not FaissIndex.load.
    static byte[] smallGraph() {
        return ringGraph(3);
    }

    private static byte[] ringGraph(int count) {
        ByteBuffer bytes = ByteBuffer.allocate(131 + count * 20).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IHNc".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(1).putLong(count).putLong(0).putLong(0).put((byte) 1).putInt(1);
        bytes.put((byte) 0).put((byte) 1).putInt(32);
        bytes.putLong(1).putDouble(1.0); // Assignment probabilities.
        bytes.putLong(2).putInt(0).putInt(2); // Cumulative degree per level.
        bytes.putLong(count);
        for (int i = 0; i < count; i++) {
            bytes.putInt(1); // Levels.
        }
        bytes.putLong(count + 1L);
        for (int i = 0; i <= count; i++) {
            bytes.putLong(i * 2L); // Offsets.
        }
        bytes.putLong(count * 2L);
        for (int i = 0; i < count; i++) {
            bytes.putInt((i + 1) % count).putInt(-1); // Neighbors.
        }
        bytes.putInt(-1).putInt(-1).putInt(16).putInt(16).putInt(0);
        bytes.put("null".getBytes(StandardCharsets.US_ASCII));
        return bytes.array();
    }

    private static void validate(byte[] bytes) throws IOException {
        validate(bytes, 300);
    }

    private static void validate(byte[] bytes, int expectedVectorCount) throws IOException {
        try (ByteArrayIndexInput input = new ByteArrayIndexInput("graph", bytes)) {
            RemoteIndexGraphValidator.validate(input, expectedVectorCount);
        }
    }

    static byte[] graphBytesWithInvalidNeighbor(String file, int neighborId) throws IOException {
        byte[] bytes = graphBytes(file, 1);
        try (ByteArrayIndexInput input = new ByteArrayIndexInput("graph", bytes)) {
            FaissHNSW graph = ((FaissHNSWProvider) FaissIndex.load(input)).getFaissHnsw();
            // Keep node 1's valid edge to node 0 so rejecting the graph cannot rely on disconnectedness.
            long offset = graph.getNeighbors().getBaseOffset() + Integer.BYTES * (graph.getOffsetsReader().get(1) + 1);
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(Math.toIntExact(offset), neighborId);
        }
        return bytes;
    }

    static byte[] graphBytesWithVectorCount(String file, long count) throws IOException {
        byte[] bytes = graphBytes(file, 1);
        String topLevelType = new String(bytes, 0, 4, StandardCharsets.US_ASCII);
        // Skip the optional ID-map header: fourcc + common float (33 bytes) or binary (21 bytes) header.
        int graphHeader = switch (topLevelType) {
            case "IxMp" -> 37;
            case "IBMp" -> 25;
            default -> 0;
        };
        String graphType = new String(bytes, graphHeader, 4, StandardCharsets.US_ASCII);
        // Binary headers have an additional int codeSize before the long vector count.
        int countOffset = graphHeader + (graphType.startsWith("IB") ? 12 : 8);
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(countOffset, count);
        return bytes;
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
