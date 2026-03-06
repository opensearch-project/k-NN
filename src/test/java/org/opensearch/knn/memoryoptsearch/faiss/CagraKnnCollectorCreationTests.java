/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.FaissHNSWTests;

/**
 * Unit tests for {@link FaissMemoryOptimizedSearcher#createKnnCollector} to verify
 * that seeded vs non-seeded strategies are handled correctly for CAGRA HNSW indices.
 */
public class CagraKnnCollectorCreationTests extends KNNTestCase {
    private static final int TOTAL_VECTORS = 300;
    private static final FlatVectorsScorer SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();

    /**
     * When the collector has a non-seeded strategy and the index is CAGRA HNSW,
     * createKnnCollector should wrap it with RandomEntryPointsKnnSearchStrategy.
     */
    @SneakyThrows
    public void testCreateKnnCollector_whenNonSeeded_thenWrapsWithRandomEntryPoints() {
        final IndexInput input = FaissHNSWTests.loadHnswBinary("data/memoryoptsearch/faiss_cagra_flat_float_300_vectors_768_dims.bin");
        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(input, null, SCORER);

        // Create a non-seeded collector
        final KnnCollector originalCollector = new TopKnnCollector(100, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);

        // We need a RandomVectorScorer for createKnnCollector
        final RandomVectorScorer scorer = new TempTestRandomVectorScorer(TOTAL_VECTORS);

        // Call createKnnCollector
        final KnnCollector result = searcher.createKnnCollector(originalCollector, scorer);

        // The result's search strategy should be RandomEntryPointsKnnSearchStrategy
        assertTrue(
            "Non-seeded collector on CAGRA should produce RandomEntryPointsKnnSearchStrategy, but got: "
                + result.getSearchStrategy().getClass().getSimpleName(),
            result.getSearchStrategy() instanceof FaissMemoryOptimizedSearcher.RandomEntryPointsKnnSearchStrategy
        );
    }

    /**
     * When the collector already has a Seeded strategy and the index is CAGRA HNSW,
     * createKnnCollector should NOT wrap it with RandomEntryPointsKnnSearchStrategy.
     * The original seeded strategy should be preserved.
     */
    @SneakyThrows
    public void testCreateKnnCollector_whenSeeded_thenPreservesOriginalStrategy() {
        final IndexInput input = FaissHNSWTests.loadHnswBinary("data/memoryoptsearch/faiss_cagra_flat_float_300_vectors_768_dims.bin");
        final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(input, null, SCORER);

        // Create a seeded strategy
        final DocIdSetIterator seedDocs = new DocIdSetIterator() {
            private int current = -1;

            @Override
            public int docID() {
                return current;
            }

            @Override
            public int nextDoc() {
                current++;
                return current >= 3 ? NO_MORE_DOCS : current * 10;
            }

            @Override
            public int advance(int target) {
                throw new UnsupportedOperationException();
            }

            @Override
            public long cost() {
                return 3;
            }
        };

        final KnnSearchStrategy seededStrategy = new KnnSearchStrategy.Seeded(seedDocs, 3, KnnSearchStrategy.Hnsw.DEFAULT);
        final KnnCollector seededCollector = new TopKnnCollector(100, Integer.MAX_VALUE, seededStrategy);

        // Get a scorer
        final RandomVectorScorer scorer = new TempTestRandomVectorScorer(TOTAL_VECTORS);

        // Call createKnnCollector
        final KnnCollector result = searcher.createKnnCollector(seededCollector, scorer);

        // The result's search strategy should NOT be RandomEntryPointsKnnSearchStrategy
        assertFalse(
            "Seeded collector on CAGRA should NOT produce RandomEntryPointsKnnSearchStrategy, but got: "
                + result.getSearchStrategy().getClass().getSimpleName(),
            result.getSearchStrategy() instanceof FaissMemoryOptimizedSearcher.RandomEntryPointsKnnSearchStrategy
        );
    }

    /**
     * Minimal RandomVectorScorer for testing createKnnCollector.
     * Only ordToDoc is needed by OrdinalTranslatedKnnCollector.
     */
    private static class TempTestRandomVectorScorer implements RandomVectorScorer {
        private final int maxOrd;

        TempTestRandomVectorScorer(int maxOrd) {
            this.maxOrd = maxOrd;
        }

        @Override
        public float score(int node) {
            return 0;
        }

        @Override
        public int maxOrd() {
            return maxOrd;
        }

        @Override
        public int ordToDoc(int ord) {
            return ord;
        }
    }
}
