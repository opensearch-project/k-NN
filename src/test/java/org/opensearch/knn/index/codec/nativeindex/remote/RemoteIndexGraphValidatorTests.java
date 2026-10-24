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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
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
            validateFixture(graphBytes(file, 1), file);
        }
    }

    public void testUnmodifiedFaissFixtures() throws IOException {
        for (String file : CAGRA_FILES) {
            try (InputStream resource = getClass().getResourceAsStream("/data/memoryoptsearch/" + file)) {
                validateFixture(resource.readAllBytes(), file);
            }
        }
    }

    public void testMissingVectorStorageIsRejectedForOrdinaryBuild() throws IOException {
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(graphBytesWithoutStorage()));
    }

    public void testCoherentGraphDimensionMustMatchBuildRequest() throws IOException {
        byte[] bytes = graphBytesWithDimension(256);
        validate(bytes, 300, 256, false);
        IOException failure = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
        assertTrue(failure.getMessage(), failure.getMessage().contains("dimension"));
    }

    public void testSkippedStorageDimensionMustMatchBuildRequest() throws IOException {
        byte[] bytes = graphBytesWithoutStorage();
        validate(bytes, 300, 768, true);
        IOException failure = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes, 300, 256, true));
        assertTrue(failure.getMessage(), failure.getMessage().contains("dimension"));
    }

    public void testBinaryDimensionMustMatchBuildRequest() throws IOException {
        byte[] bytes = binaryGraphWithSkippedStorage(1);
        validate(bytes, 3, 8, VectorDataType.BINARY, SpaceType.HAMMING, true);
        IOException failure = expectThrows(
            RemoteIndexGraphValidator.GraphFaultException.class,
            () -> validate(bytes, 3, 16, VectorDataType.BINARY, SpaceType.HAMMING, true)
        );
        assertTrue(failure.getMessage(), failure.getMessage().contains("dimension"));
    }

    public void testSkippedCagraStorageIsAcceptedWhenRequested() throws IOException {
        validate(graphBytesWithoutStorage(), 300, 768, true);
    }

    public void testEmbeddedStorageIsRejectedWhenSkippedStorageWasRequested() throws IOException {
        expectThrows(
            RemoteIndexGraphValidator.GraphFaultException.class,
            () -> validate(graphBytes(CAGRA_FILES.getFirst(), 1), 300, 768, true)
        );
    }

    public void testPlainHnswCannotUseSkippedStorage() {
        byte[] cagra = smallGraphWithUpperLevel(0);
        ByteBuffer graph = ByteBuffer.allocate(cagra.length - 6);
        graph.put("IHNf".getBytes(StandardCharsets.US_ASCII)).put(cagra, 4, 33).put(cagra, 43, cagra.length - 43);
        IOException failure = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(graph.array(), 3));
        assertTrue(failure.getMessage(), failure.getMessage().contains("skipped vector storage"));
    }

    public void testMissingIdMapIsRejected() throws IOException {
        byte[] bytes = graphBytes(CAGRA_FILES.getFirst(), 1);
        // Retain a complete CAGRA graph and its flat storage, removing only the ID-map header and trailing IDs.
        byte[] graph = Arrays.copyOfRange(bytes, 37, bytes.length - Long.BYTES * 301);
        IOException failure = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(graph));
        assertTrue(failure.getMessage(), failure.getMessage().contains("document ID map"));
    }

    public void testNonBaseCagraRequiresEntryPoint() throws IOException {
        byte[] bytes = graphBytes(CAGRA_FILES.getFirst(), 1);
        // IxMp header (37 bytes), then CAGRA fourcc/common header (37 bytes) and two flags.
        bytes[75] = 0; // base_level_only=false; this fixture has entry_point=max_level=-1.
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
    }

    public void testNonBaseCagraChecksMaximumLevelAndEntryPointLevel() {
        for (int entryPoint : new int[] { -1, 1, 3, Integer.MAX_VALUE }) {
            byte[] graph = smallGraphWithUpperLevel(0);
            graph[38] = 0;
            ByteBuffer.wrap(graph).order(ByteOrder.LITTLE_ENDIAN).putInt(175, entryPoint);
            expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(graph, 3));
        }
        for (int maxLevel : new int[] { -1, 0, 2, Integer.MAX_VALUE }) {
            byte[] graph = smallGraphWithUpperLevel(0);
            graph[38] = 0;
            ByteBuffer.wrap(graph).order(ByteOrder.LITTLE_ENDIAN).putInt(179, maxLevel);
            expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(graph, 3));
        }
    }

    public void testNonBaseCagraWithValidMetadataIsAccepted() throws IOException {
        byte[] graph = smallGraphWithUpperLevel(0);
        graph[38] = 0;
        validateGraph(graph, 3);
    }

    public void testBaseOnlyCagraRejectsUnsafeRangeSearchMetadata() {
        for (int entryPoint : new int[] { -2, 1, 3, Integer.MAX_VALUE }) {
            byte[] graph = smallGraphWithUpperLevel(0);
            ByteBuffer.wrap(graph).order(ByteOrder.LITTLE_ENDIAN).putInt(175, entryPoint);
            expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(graph, 3));
        }
        for (int maxLevel : new int[] { -2, 2, Integer.MAX_VALUE }) {
            byte[] graph = smallGraphWithUpperLevel(0);
            ByteBuffer.wrap(graph).order(ByteOrder.LITTLE_ENDIAN).putInt(179, maxLevel);
            expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(graph, 3));
        }
        byte[] graph = smallGraphWithUpperLevel(0);
        // Native range search uses a random seed if entry_point=-1. Nodes 1 and 2 do not have level 1.
        ByteBuffer.wrap(graph).order(ByteOrder.LITTLE_ENDIAN).putInt(175, -1);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(graph, 3));
    }

    public void testBaseOnlyCagraCanSearchBelowHighestStoredLevel() throws IOException {
        for (int entryPoint : new int[] { -1, 0, 1, 2 }) {
            for (int maxLevel : new int[] { -1, 0 }) {
                byte[] graph = smallGraphWithUpperLevel(0);
                ByteBuffer.wrap(graph).order(ByteOrder.LITTLE_ENDIAN).putInt(175, entryPoint).putInt(179, maxLevel);
                validateGraph(graph, 3);
            }
        }
    }

    public void testBaseOnlyCagraCanUseRandomSeedsWhenEveryNodeHasUpperLevel() throws IOException {
        ByteBuffer graph = ByteBuffer.allocate(207).order(ByteOrder.LITTLE_ENDIAN);
        graph.put("IHNc".getBytes(StandardCharsets.US_ASCII));
        graph.putInt(1).putLong(3).putLong(0).putLong(0).put((byte) 1).putInt(1);
        graph.put((byte) 0).put((byte) 1).putInt(32);
        graph.putLong(1).putDouble(1.0);
        graph.putLong(3).putInt(0).putInt(2).putInt(3);
        graph.putLong(3).putInt(2).putInt(2).putInt(2);
        graph.putLong(4).putLong(0).putLong(3).putLong(6).putLong(9);
        graph.putLong(9);
        for (int node = 0; node < 3; node++) {
            graph.putInt((node + 1) % 3).putInt(-1).putInt(node);
        }
        graph.putInt(-1).putInt(1).putInt(16).putInt(16).putInt(0);
        graph.put("null".getBytes(StandardCharsets.US_ASCII));
        validateGraph(graph.array(), 3);
    }

    public void testNonBaseCagraStillNeedsRandomEntryPointsForMemoryOptimizedSearch() {
        byte[] graph = smallGraphWithUpperLevel(0);
        graph[38] = 0;
        ByteBuffer.wrap(graph).order(ByteOrder.LITTLE_ENDIAN).putInt(39, 0);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(graph, 3));
    }

    public void testSkippedBinaryStorageAcceptsAdcMetrics() throws IOException {
        for (int metric : new int[] { 0, 1 }) {
            validate(binaryGraphWithSkippedStorage(metric), 3, 8, VectorDataType.BINARY, SpaceType.HAMMING, true);
        }
    }

    public void testSkippedBinaryStorageRejectsUnsupportedAdcMetric() {
        expectThrows(
            RemoteIndexGraphValidator.GraphFaultException.class,
            () -> validate(binaryGraphWithSkippedStorage(2), 3, 8, VectorDataType.BINARY, SpaceType.HAMMING, true)
        );
    }

    private static byte[] binaryGraphWithSkippedStorage(int metric) {
        byte[] graph = smallGraphWithUpperLevel(0);
        ByteBuffer bytes = ByteBuffer.allocate(25 + 25 + graph.length - 43 + 8 + 24).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IBMp".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(8).putInt(1).putLong(3).put((byte) 1).putInt(metric);
        bytes.put("IBHf".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(8).putInt(1).putLong(3).put((byte) 1).putInt(1);
        bytes.put(graph, 43, graph.length - 43).putLong(3).putLong(0).putLong(1).putLong(2);
        return bytes.array();
    }

    public void testCagraFlagsMustBeBoolean() {
        for (int offset : new int[] { 37, 38 }) {
            byte[] graph = smallGraph();
            graph[offset] = 2;
            expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(graph, 3));
        }
    }

    public void testFloatIdMapMetricMustMatchGraph() throws IOException {
        assertMetricMismatchRejected(33);
    }

    public void testFloatGraphMetricMustMatchIdMap() throws IOException {
        assertMetricMismatchRejected(70);
    }

    public void testFloatStorageMetricMustMatchGraph() throws IOException {
        assertMetricMismatchRejected(43336 + 33);
    }

    public void testConsistentInnerProductHeadersAreAccepted() throws IOException {
        byte[] bytes = graphBytes(CAGRA_FILES.getFirst(), 1);
        ByteBuffer data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        data.putInt(33, 0).putInt(70, 0).putInt(43336 + 33, 0);
        data.position(43336);
        data.put("IxFI".getBytes(StandardCharsets.US_ASCII));
        validate(bytes, 300, 768, VectorDataType.FLOAT, SpaceType.INNER_PRODUCT, false);
    }

    public void testBinaryIdMapCanRecordDifferentMetric() throws IOException {
        byte[] bytes = graphBytes("faiss_cagra2_flat_binary_300_vectors_768_dims.bin", 1);
        assertEquals("IBMp", new String(bytes, 0, 4, StandardCharsets.US_ASCII));
        // Binary ID maps may record the ADC metric independently from the graph's default Hamming metric.
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(21, 0);
        validate(bytes, 300, 768, VectorDataType.BINARY, SpaceType.HAMMING, false);
    }

    private void assertMetricMismatchRejected(int metricOffset) throws IOException {
        byte[] bytes = graphBytes(CAGRA_FILES.getFirst(), 1);
        ByteBuffer data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(1, data.getInt(metricOffset));
        data.putInt(metricOffset, 0);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
    }

    static byte[] graphBytesWithoutStorage() throws IOException {
        byte[] original = graphBytes(CAGRA_FILES.getFirst(), 1);
        // Layout of the independent 300-vector FP32 fixture: graph ends at 43336; ID map is the last counted long array.
        int storageOffset = 43336;
        assertEquals("IxF2", new String(original, storageOffset, 4, StandardCharsets.US_ASCII));
        int idMapOffset = original.length - Long.BYTES * 301;
        ByteBuffer bytes = ByteBuffer.allocate(storageOffset + 4 + original.length - idMapOffset);
        bytes.put(original, 0, storageOffset).put("null".getBytes(StandardCharsets.US_ASCII));
        bytes.put(original, idMapOffset, original.length - idMapOffset);
        return bytes.array();
    }

    public void testDisconnectedCagraGraphs() throws IOException {
        for (String file : CAGRA_FILES) {
            IOException exception = expectThrows(
                RemoteIndexGraphValidator.GraphFaultException.class,
                () -> validateFixture(graphBytes(file, 2), file)
            );
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
        validate(withIdMap(buffer.array(), 8, 1), 1, 8, true);
    }

    public void testInvalidNeighbor() throws IOException {
        byte[] bytes = graphBytesWithInvalidNeighbor(CAGRA_FILES.getFirst(), 300);
        IOException exception = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes));
        assertTrue(exception.getMessage(), exception.getMessage().contains("invalid neighbor ID"));
    }

    public void testInvalidNeighborMatchingLuceneSentinel() throws IOException {
        for (String file : CAGRA_FILES) {
            byte[] bytes = graphBytesWithInvalidNeighbor(file, Integer.MAX_VALUE);
            IOException exception = expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateFixture(bytes, file));
            assertTrue(exception.getMessage(), exception.getMessage().contains("invalid neighbor ID"));
        }
    }

    public void testInvalidGraphVectorCounts() throws IOException {
        for (String file : CAGRA_FILES) {
            for (long count : new long[] { -1, 0, 299, Integer.MAX_VALUE, Long.MAX_VALUE }) {
                byte[] bytes = graphBytesWithVectorCount(file, count);
                expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateFixture(bytes, file));
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
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes, 299, 768, false));
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
            assertSame(
                failure,
                expectThrows(
                    IOException.class,
                    () -> RemoteIndexGraphValidator.validate(
                        input,
                        300,
                        768,
                        300,
                        VectorDataType.FLOAT.getValue(),
                        SpaceType.L2.getValue(),
                        false
                    )
                )
            );
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
            assertSame(
                failure,
                expectThrows(
                    IOException.class,
                    () -> RemoteIndexGraphValidator.validate(
                        input,
                        300,
                        768,
                        300,
                        VectorDataType.FLOAT.getValue(),
                        SpaceType.L2.getValue(),
                        false
                    )
                )
            );
        }
    }

    public void testHandWrittenCagraGraph() throws IOException {
        validateGraph(smallGraph(), 3);
    }

    public void testLargeSerializedGraph() throws IOException {
        validateGraph(ringGraph(100_000), 100_000);
    }

    public void testOversizedCumulativeTableIsRejectedBeforeAllocation() {
        byte[] bytes = smallGraph();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(59, Integer.MAX_VALUE);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(bytes, 3));
    }

    public void testOversizedDegreeIsRejectedBeforeAllocation() {
        byte[] bytes = smallGraph();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(71, Integer.MAX_VALUE);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(bytes, 3));
    }

    public void testTerminalOffsetMustMatchNeighborSection() {
        byte[] bytes = smallGraph();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(127, 7);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(bytes, 3));
    }

    public void testIdMapMustContainAllVectorIds() {
        byte[] graph = smallGraph();
        ByteBuffer bytes = ByteBuffer.allocate(37 + graph.length + 8).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IxMp".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(1).putLong(3).putLong(0).putLong(0).put((byte) 1).putInt(1);
        bytes.put(graph).putLong(0);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(bytes.array(), 3, 1, true));
    }

    public void testNestedGraphCannotBeUsedAsFlatStorage() {
        byte[] graph = smallGraph();
        ByteBuffer bytes = ByteBuffer.allocate(graph.length * 2 - 4);
        bytes.put(graph, 0, graph.length - 4).put(graph);
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(withIdMap(bytes.array(), 1, 3), 3, 1, false));
    }

    public void testValidationHonorsMergeCancellation() throws Exception {
        try (MockedStatic<MergeAbortChecker> mergeAbort = mockStatic(MergeAbortChecker.class)) {
            mergeAbort.when(MergeAbortChecker::isMergeAborted).thenReturn(true);
            expectThrows(IndexBuildAbortedException.class, () -> validateGraph(smallGraph(), 3));
        }
    }

    public void testValidationHonorsCancellationDuringScan() throws Exception {
        try (MockedStatic<MergeAbortChecker> mergeAbort = mockStatic(MergeAbortChecker.class)) {
            mergeAbort.when(MergeAbortChecker::isMergeAborted).thenReturn(false, false, false, true);
            expectThrows(IndexBuildAbortedException.class, () -> validateGraph(ringGraph(2048), 2048));
        }
    }

    public void testCancellationPollingAccumulatesNeighborWork() throws Exception {
        int count = 300;
        byte[] bytes = withIdMap(ringGraph(count), 1, count);
        try (ByteArrayIndexInput input = new ByteArrayIndexInput("graph", bytes)) {
            FaissHNSW graph = ((FaissHNSWProvider) FaissIndex.load(input)).getFaissHnsw();
            long invalidNeighborOffset = graph.getNeighbors().getBaseOffset() + Integer.BYTES * graph.getOffsetsReader().get(count - 1);
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(Math.toIntExact(invalidNeighborOffset), count);
        }
        try (MockedStatic<MergeAbortChecker> mergeAbort = mockStatic(MergeAbortChecker.class)) {
            // The fifth poll is reached only when node, level, and neighbor work shares one monotonic counter. A
            // per-loop counter reaches the deliberately invalid final neighbor first and reports a graph fault instead.
            mergeAbort.when(MergeAbortChecker::isMergeAborted).thenReturn(false, false, false, false, true);
            expectThrows(IndexBuildAbortedException.class, () -> validate(bytes, count, 1, true));
        }
    }

    public void testEveryTruncatedPrefixIsRejected() {
        byte[] graph = withIdMap(smallGraph(), 1, 3);
        for (int length = 0; length < graph.length; length++) {
            byte[] prefix = Arrays.copyOf(graph, length);
            expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validate(prefix, 3, 1, true));
        }
    }

    public void testSectionLengthOverflowIsRejected() {
        for (int offset : new int[] { 43, 59, 75, 95, 135 }) {
            for (long count : new long[] { -1, Long.MIN_VALUE, Long.MAX_VALUE, 1L << 62 }) {
                byte[] bytes = smallGraph();
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(offset, count);
                expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(bytes, 3));
            }
        }
    }

    public void testUpperLevelNeighborMustExistOnThatLevel() {
        expectThrows(RemoteIndexGraphValidator.GraphFaultException.class, () -> validateGraph(smallGraphWithUpperLevel(1), 3));
    }

    public void testValidUpperLevelGraph() throws IOException {
        validateGraph(smallGraphWithUpperLevel(0), 3);
    }

    public void testValidThreeLevelGraph() throws IOException {
        validateGraph(smallGraphWithThreeLevels(), 3);
    }

    public void testHnswEntryPointMustExistAtMaximumLevel() throws IOException {
        byte[] cagra = smallGraphWithUpperLevel(0);
        // Plain HNSW has the same graph sections without the six CAGRA-specific header bytes.
        // Replace the four-byte "null" with a 37-byte flat header, vector count and three float values.
        ByteBuffer bytes = ByteBuffer.allocate(cagra.length - 6 - 4 + 37 + 8 + 12).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IHNf".getBytes(StandardCharsets.US_ASCII));
        bytes.put(cagra, 4, 33).put(cagra, 43, cagra.length - 43 - 4);
        bytes.put("IxF2".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(1).putLong(3).putLong(0).putLong(0).put((byte) 1).putInt(1);
        bytes.putLong(3).putFloat(0).putFloat(1).putFloat(2);
        validate(withIdMap(bytes.array(), 1, 3), 3, 1, false);
        bytes.putInt(169, 1); // Node 1 only belongs to level 0; maximum level is 1.
        IOException failure = expectThrows(
            RemoteIndexGraphValidator.GraphFaultException.class,
            () -> validate(withIdMap(bytes.array(), 1, 3), 3, 1, false)
        );
        assertTrue(failure.getMessage(), failure.getMessage().contains("maximum level"));
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

    private static byte[] smallGraphWithThreeLevels() {
        ByteBuffer bytes = ByteBuffer.allocate(207).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IHNc".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(1).putLong(3).putLong(0).putLong(0).put((byte) 1).putInt(1);
        bytes.put((byte) 0).put((byte) 0).putInt(32);
        bytes.putLong(1).putDouble(1.0);
        bytes.putLong(4).putInt(0).putInt(2).putInt(3).putInt(4);
        bytes.putLong(3).putInt(3).putInt(1).putInt(1);
        bytes.putLong(4).putLong(0).putLong(4).putLong(6).putLong(8);
        bytes.putLong(8).putInt(1).putInt(-1).putInt(0).putInt(0).putInt(2).putInt(-1).putInt(0).putInt(-1);
        bytes.putInt(0).putInt(2).putInt(16).putInt(16).putInt(0);
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

    static byte[] graphBytesWithDimension(int dimension) throws IOException {
        byte[] original = graphBytes(CAGRA_FILES.getFirst(), 1);
        int storageOffset = 43336;
        int vectorCountOffset = storageOffset + 37;
        int vectorsOffset = vectorCountOffset + Long.BYTES;
        int idsOffset = original.length - Long.BYTES * 301;
        int vectorBytes = dimension * 300 * Float.BYTES;
        ByteBuffer resized = ByteBuffer.allocate(vectorsOffset + vectorBytes + original.length - idsOffset).order(ByteOrder.LITTLE_ENDIAN);
        resized.put(original, 0, vectorCountOffset);
        resized.putLong((long) dimension * 300);
        resized.put(original, vectorsOffset, vectorBytes);
        resized.put(original, idsOffset, original.length - idsOffset);
        // Change all three headers and the flat vector payload together, leaving a coherent index of the wrong dimension.
        resized.putInt(4, dimension).putInt(41, dimension).putInt(storageOffset + 4, dimension);
        return resized.array();
    }

    static byte[] graphBytesWithMetric(int metric) throws IOException {
        return graphBytesWithMetric(CAGRA_FILES.getFirst(), metric);
    }

    static byte[] graphBytesWithMetric(String file, int metric) throws IOException {
        byte[] bytes = graphBytes(file, 1);
        ByteBuffer data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        data.putInt(33, metric).putInt(70, metric);
        for (int offset = 0; offset <= bytes.length - 4; offset++) {
            if (bytes[offset] == 'I' && bytes[offset + 1] == 'x' && bytes[offset + 2] == 'S' && bytes[offset + 3] == 'Q') {
                data.putInt(offset + 33, metric);
                return bytes;
            }
            if (bytes[offset] == 'I' && bytes[offset + 1] == 'x' && bytes[offset + 2] == 'F') {
                data.putInt(offset + 33, metric);
                return bytes;
            }
        }
        throw new AssertionError("Fixture has no scalar or flat vector storage");
    }

    static byte[] graphBytesWithLastDocumentId(long documentId) throws IOException {
        byte[] bytes = graphBytes(CAGRA_FILES.getFirst(), 1);
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(bytes.length - Long.BYTES, documentId);
        return bytes;
    }

    private static void validate(byte[] bytes) throws IOException {
        validate(bytes, 300, 768, false);
    }

    private static void validateFixture(byte[] bytes, String file) throws IOException {
        boolean binary = file.contains("binary");
        VectorDataType vectorDataType = file.contains("flat_byte") ? VectorDataType.BYTE
            : file.contains("flat_fp16") ? VectorDataType.HALF_FLOAT
            : binary ? VectorDataType.BINARY
            : VectorDataType.FLOAT;
        SpaceType spaceType = binary ? SpaceType.HAMMING
            : file.equals("faiss_cagra_flat_float_300_vectors_768_dims.bin") ? SpaceType.L2
            : SpaceType.INNER_PRODUCT;
        validate(bytes, 300, file.equals("faiss_binary_cagra_300_vectors_32_dims.faiss") ? 32 : 768, vectorDataType, spaceType, false);
    }

    private static void validate(byte[] bytes, int expectedVectorCount, int expectedDimension, boolean skipStoredVectors)
        throws IOException {
        validate(bytes, expectedVectorCount, expectedDimension, VectorDataType.FLOAT, SpaceType.L2, skipStoredVectors);
    }

    private static void validate(
        byte[] bytes,
        int expectedVectorCount,
        int expectedDimension,
        VectorDataType vectorDataType,
        SpaceType spaceType,
        boolean skipStoredVectors
    ) throws IOException {
        try (ByteArrayIndexInput input = new ByteArrayIndexInput("graph", bytes)) {
            RemoteIndexGraphValidator.validate(
                input,
                expectedVectorCount,
                expectedDimension,
                expectedVectorCount,
                vectorDataType.getValue(),
                spaceType.getValue(),
                skipStoredVectors
            );
        }
    }

    private static void validateGraph(byte[] graph, int count) throws IOException {
        validate(withIdMap(graph, 1, count), count, 1, true);
    }

    private static byte[] withIdMap(byte[] graph, int dimension, int count) {
        ByteBuffer bytes = ByteBuffer.allocate(37 + graph.length + Long.BYTES * (count + 1)).order(ByteOrder.LITTLE_ENDIAN);
        bytes.put("IxMp".getBytes(StandardCharsets.US_ASCII));
        bytes.putInt(dimension).putLong(count).putLong(0).putLong(0).put((byte) 1).putInt(1);
        bytes.put(graph).putLong(count);
        for (int i = 0; i < count; i++) {
            bytes.putLong(i);
        }
        return bytes.array();
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
