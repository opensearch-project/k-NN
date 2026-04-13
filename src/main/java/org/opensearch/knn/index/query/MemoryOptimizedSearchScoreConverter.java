/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

/**
 * Utility class for converting between Faiss and Lucene score representations
 * in memory-optimized search.
 *
 * <p>Memory-optimized search runs Lucene on top of a Faiss index. It leverages
 * Lucene's efficient algorithms and Lucene's {@code Directory} architecture for efficient loading to
 * produce the same results as when memory optimization is disabled.</p>
 * With the same query, results are expected to be identical regardless of
 * whether memory optimization is enabled.
 *
 * <p>Note that this should be used only when memory_optimized_search is enabled.
 */
public final class MemoryOptimizedSearchScoreConverter {
    /**
     * Convert Faiss distance to Lucene radial threshold.
     *
     * @param distance Faiss distance
     * @param spaceType Space type being used.
     * @return Converted value to be used during Lucene search algorithm.
     */
    public static float distanceToRadialThreshold(final float distance, final SpaceType spaceType) {
        return KNNEngine.LUCENE.distanceToRadialThreshold(distance, spaceType);
    }

    /**
     * Convert Faiss score to Lucene radial threshold.
     *
     * @param score Faiss score
     * @param spaceType Space type that's being used
     * @return Converted radial threshold for Lucene
     */
    public static float scoreToRadialThreshold(final float score, final SpaceType spaceType) {
        return KNNEngine.LUCENE.scoreToRadialThreshold(score, spaceType);
    }
}
