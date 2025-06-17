/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;

public class RandomEntryPointsKnnSearchStrategyTests {
    @Test
    @SneakyThrows
    public void randomEntryPointsGenerationTests() {
        final int numEntries = 100;
        final long numVectors = 10000;

        // Create strategy
        final FaissMemoryOptimizedSearcher.RandomEntryPointsKnnSearchStrategy strategy =
            new FaissMemoryOptimizedSearcher.RandomEntryPointsKnnSearchStrategy(numEntries, numVectors, mock(KnnSearchStrategy.class));

        // Validate #entry points
        assertEquals(numEntries, strategy.numberOfEntryPoints());

        // We should get exactly `numEntries` vector ids.
        final DocIdSetIterator iterator = strategy.entryPoints();
        for (int i = 0; i < numEntries; ++i) {
            final int internalVectorId = iterator.nextDoc();
            assertTrue(internalVectorId >= 0);
            assertTrue(internalVectorId < numVectors);
        }
    }
}
