/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight;
import org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector;

import java.lang.reflect.Field;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_DECAY;

public class MemoryOptimizedKNNWeightTests extends KNNTestCase {

    public void testAcornIsDisabled() throws Exception {
        Field field = MemoryOptimizedKNNWeight.class.getDeclaredField("DEFAULT_HNSW_SEARCH_STRATEGY");
        field.setAccessible(true);
        KnnSearchStrategy.Hnsw strategy = (KnnSearchStrategy.Hnsw) field.get(null);

        assertEquals(
            "ACORN threshold must be 0 to disable filtered search and match Lucene 10.4 behavior",
            0,
            strategy.filteredSearchThreshold()
        );
        assertFalse("useFilteredSearch must return false for any filtering rate when threshold is 0", strategy.useFilteredSearch(0.5f));
    }

    // Validates that the memory-optimized search (MOS) radius path builds the decay-based
    // RadiusVectorSimilarityCollector and wires in the shared decay factor
    // (DEFAULT_LUCENE_RADIAL_SEARCH_DECAY = 0.95). A radius (r-NN) query is simulated by passing k = null,
    // which routes the constructor to the radius branch that builds the collector manager.
    public void testRadiusSearch_usesDecayBasedCollector() throws Exception {
        final KNNQuery knnQuery = mock(KNNQuery.class);
        when(knnQuery.getRadius()).thenReturn(0.5f);
        final IndexSearcher searcher = mock(IndexSearcher.class);

        // k == null -> radius search branch, which creates the RadiusVectorSimilarityCollector manager.
        final MemoryOptimizedKNNWeight weight = new MemoryOptimizedKNNWeight(knnQuery, 1.0f, null, searcher, null);

        // Reach the private collector manager built for the radius search path.
        final Field managerField = MemoryOptimizedKNNWeight.class.getDeclaredField("knnCollectorManager");
        managerField.setAccessible(true);
        final KnnCollectorManager manager = (KnnCollectorManager) managerField.get(weight);

        // The radius lambda ignores the search strategy and context, so null is acceptable here.
        final KnnCollector collector = manager.newCollector(Integer.MAX_VALUE, null, null);
        assertTrue(
            "MOS radius search must use the decay-based RadiusVectorSimilarityCollector",
            collector instanceof RadiusVectorSimilarityCollector
        );

        // Verify the decay factor wired into the collector is the shared default (0.95).
        final Field decayField = RadiusVectorSimilarityCollector.class.getDeclaredField("decay");
        decayField.setAccessible(true);
        assertEquals(DEFAULT_LUCENE_RADIAL_SEARCH_DECAY, (float) decayField.get(collector), 0.0f);
    }
}
