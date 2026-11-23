/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.cagra.FaissCagraHNSW;

import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public class FaissCagraHNSWTests extends KNNTestCase {
    // HNSW graph section having two levels. entry_point=12, max_level=1
    private static final String HIERARCHICAL_HNSW = "data/memoryoptsearch/faiss_hnsw_100_vectors.bin";
    private static final int HIERARCHICAL_HNSW_NUM_VECTORS = 100;
    private static final int HIERARCHICAL_HNSW_ENTRY_POINT = 12;
    private static final int HIERARCHICAL_HNSW_MAX_LEVEL = 1;

    // HNSW graph section having the bottom level only. entry_point=0, max_level=0
    private static final String SINGLE_LEVEL_HNSW = "data/memoryoptsearch/faiss_hnsw_one_vector.bin";

    @SneakyThrows
    public void testLoad_whenBaseLevelOnly_thenOverridesEntryPointAndMaxLevel() {
        final FaissCagraHNSW cagraHnsw = new FaissCagraHNSW();
        cagraHnsw.load(loadHnswBinary(HIERARCHICAL_HNSW), HIERARCHICAL_HNSW_NUM_VECTORS, true);

        // Even though the graph has a hierarchy, the flag should be honored.
        assertBaseLevelOnly(cagraHnsw);
    }

    @SneakyThrows
    public void testLoad_whenBaseLevelOnlyFlagIsNotSetAndHasUpperLevels_thenKeepsEntryPointAndMaxLevel() {
        final FaissCagraHNSW cagraHnsw = new FaissCagraHNSW();
        cagraHnsw.load(loadHnswBinary(HIERARCHICAL_HNSW), HIERARCHICAL_HNSW_NUM_VECTORS, false);

        assertFalse(cagraHnsw.isBaseLevelOnly());
        assertEquals(HIERARCHICAL_HNSW_ENTRY_POINT, cagraHnsw.getEntryPoint());
        assertEquals(HIERARCHICAL_HNSW_MAX_LEVEL, cagraHnsw.getMaxLevel());
        assertEquals(HIERARCHICAL_HNSW_NUM_VECTORS, cagraHnsw.getTotalNumberOfVectors());
    }

    @SneakyThrows
    public void testLoad_whenBaseLevelOnlyFlagIsNotSetButNoUpperLevels_thenFallsBackToBaseLevelOnly() {
        final FaissCagraHNSW cagraHnsw = new FaissCagraHNSW();
        cagraHnsw.load(loadHnswBinary(SINGLE_LEVEL_HNSW), 1, false);

        // The graph has a valid entry point, but no upper levels to descend.
        assertBaseLevelOnly(cagraHnsw);
    }

    @SneakyThrows
    public void testLoad_whenFlagIsNotGiven_thenBaseLevelOnly() {
        // Via subtype
        final FaissCagraHNSW cagraHnsw = new FaissCagraHNSW();
        cagraHnsw.load(loadHnswBinary(HIERARCHICAL_HNSW), HIERARCHICAL_HNSW_NUM_VECTORS);
        assertBaseLevelOnly(cagraHnsw);

        // Via base type
        final FaissHNSW hnsw = new FaissCagraHNSW();
        hnsw.load(loadHnswBinary(HIERARCHICAL_HNSW), HIERARCHICAL_HNSW_NUM_VECTORS);
        assertBaseLevelOnly((FaissCagraHNSW) hnsw);
    }

    public void testLoadIndexFile_whenBaseLevelOnlyFlagIsSet_thenBaseLevelOnly() {
        // IHNc, FP32
        assertBaseLevelOnly(loadCagraHnsw("data/memoryoptsearch/faiss_cagra_flat_float_300_vectors_768_dims.bin"));
        // IHc2, FP16
        assertBaseLevelOnly(loadCagraHnsw("data/memoryoptsearch/faiss_cagra2_flat_fp16_300_vectors_768_dims.bin"));
        assertBaseLevelOnly(loadCagraHnsw("data/memoryoptsearch/faiss_cagra2_flat_bf16_300_vectors_768_dims.bin"));
        // IBHc, binary
        assertBaseLevelOnly(loadCagraHnsw("data/memoryoptsearch/faiss_binary_cagra_300_vectors_32_dims.faiss"));
    }

    public void testLoadIndexFile_whenBaseLevelOnlyFlagIsNotSetAndHasUpperLevels_thenMultiLayer() {
        // IHc2, FP32
        doTestMultiLayerIndexFile("data/memoryoptsearch/faiss_cagra2_multilayer_flat_float_300_vectors_768_dims.bin");
        // IBHc, binary
        doTestMultiLayerIndexFile("data/memoryoptsearch/faiss_cagra2_flat_binary_300_vectors_768_dims.bin");
    }

    private static void doTestMultiLayerIndexFile(final String relativePath) {
        final FaissCagraHNSW cagraHnsw = loadCagraHnsw(relativePath);
        assertFalse(relativePath, cagraHnsw.isBaseLevelOnly());
        assertTrue(relativePath, cagraHnsw.getEntryPoint() >= 0);
        assertTrue(relativePath, cagraHnsw.getEntryPoint() < cagraHnsw.getTotalNumberOfVectors());
        assertTrue(relativePath, cagraHnsw.getMaxLevel() >= 1);
    }

    private static void assertBaseLevelOnly(final FaissCagraHNSW cagraHnsw) {
        assertTrue(cagraHnsw.isBaseLevelOnly());
        // Entry points are provided at search time.
        assertEquals(-1, cagraHnsw.getEntryPoint());
        // 0-based index of the top level, 0 indicates it has the bottom level only.
        assertEquals(0, cagraHnsw.getMaxLevel());
    }

    @SneakyThrows
    private static FaissCagraHNSW loadCagraHnsw(final String relativePath) {
        final IndexInput input = loadHnswBinary(relativePath);
        final FaissIndex faissIndex = FaissIndex.load(input);
        assertTrue(relativePath, faissIndex instanceof FaissIdMapIndex);
        final FaissHNSW hnsw = ((FaissIdMapIndex) faissIndex).getFaissHnsw();
        assertTrue(relativePath, hnsw instanceof FaissCagraHNSW);
        return (FaissCagraHNSW) hnsw;
    }
}
