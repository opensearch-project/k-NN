/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight;

import java.lang.reflect.Field;

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
}
