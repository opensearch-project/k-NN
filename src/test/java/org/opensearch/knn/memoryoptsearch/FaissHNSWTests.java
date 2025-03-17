/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;

public class FaissHNSWTests extends KNNTestCase {
    @SneakyThrows
    public void testLoadGraphWithSingleVector() {
        final IndexInput indexInput = loadHnswBinary("data/memoryoptsearch/faiss_hnsw_one_vector.bin");
        final FaissHNSW faissHNSW = new FaissHNSW();
        faissHNSW.load(indexInput, 1);
        doTest(faissHNSW, new int[] { 0, 32, 48, 64, 80, 96, 112, 128, 144 }, new long[] { 0, 32 }, 160, 128, 0, 0, 100);
    }

    @SneakyThrows
    public void testLoadGraphWithNVectors() {
        final IndexInput indexInput = loadHnswBinary("data/memoryoptsearch/faiss_hnsw_100_vectors.bin");
        final FaissHNSW faissHNSW = new FaissHNSW();
        faissHNSW.load(indexInput, 100);
        final int[] cumulativeNumNeighbors = new int[] { 0, 32, 48, 64, 80, 96, 112, 128, 144 };
        doTest(faissHNSW, cumulativeNumNeighbors, ANSWER_OFFSETS, 1348, 13184, 12, 1, 100);
    }

    @SneakyThrows
    public static IndexInput loadHnswBinary(final String relativePath) {
        final URL hnswWithOneVector = FaissHNSWTests.class.getClassLoader().getResource(relativePath);
        final byte[] bytes = Files.readAllBytes(Path.of(hnswWithOneVector.toURI()));
        final IndexInput indexInput = new ByteArrayIndexInput("FaissHNSWTests", bytes);
        return indexInput;
    }

    private void doTest(
        final FaissHNSW faissHNSW,
        final int[] cumulativeNumNeighbors,
        final long[] offsets,
        final long neighborsBaseOffset,
        final long neighborsSectionSize,
        final int entryPoint,
        final int maxLevel,
        final int efSearch
    ) {
        // Cumulative number of neighbor per level
        assertArrayEquals(cumulativeNumNeighbors, faissHNSW.getCumNumberNeighborPerLevel());

        // offsets
        final DirectMonotonicReader offsetsReader = faissHNSW.getOffsetsReader();
        for (int i = 0; i < offsets.length; i++) {
            assertEquals(offsets[i], offsetsReader.get(i));
        }

        // neighbors
        assertEquals(neighborsBaseOffset, faissHNSW.getNeighbors().getBaseOffset());
        assertEquals(neighborsSectionSize, faissHNSW.getNeighbors().getSectionSize());

        // entry point
        assertEquals(entryPoint, faissHNSW.getEntryPoint());

        // max level
        assertEquals(maxLevel, faissHNSW.getMaxLevel());

        // efSearch
        assertEquals(efSearch, faissHNSW.getEfSearch());
    }

    private static final long[] ANSWER_OFFSETS = new long[] {
        0,
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        400,
        448,
        480,
        512,
        544,
        576,
        608,
        640,
        672,
        704,
        736,
        784,
        816,
        848,
        880,
        912,
        944,
        976,
        1008,
        1040,
        1072,
        1104,
        1136,
        1168,
        1200,
        1248,
        1280,
        1312,
        1344,
        1376,
        1408,
        1440,
        1472,
        1504,
        1536,
        1568,
        1600,
        1632,
        1664,
        1696,
        1728,
        1776,
        1808,
        1840,
        1872,
        1904,
        1936,
        1968,
        2000,
        2032,
        2064,
        2096,
        2128,
        2160,
        2192,
        2224,
        2256,
        2288,
        2320,
        2352,
        2384,
        2416,
        2464,
        2496,
        2528,
        2560,
        2592,
        2624,
        2656,
        2688,
        2720,
        2752,
        2784,
        2816,
        2848,
        2880,
        2912,
        2944,
        2976,
        3008,
        3040,
        3072,
        3104,
        3136,
        3168,
        3200,
        3232,
        3264,
        3296 };
}
