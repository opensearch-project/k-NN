/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.ScoreDoc;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

/**
 * Utility class for converting between Faiss and Lucene score representations
 * in memory-optimized search.
 *
 * <p>Memory-optimized search runs Lucene on top of a Faiss index. It leverages
 * Lucene’s efficient algorithms and Lucene’s {@code Directory} architecture for efficient loading to
 * produce the same results as when memory optimization is disabled.</p>
 * With the same query, results are expected to be identical regardless of
 * whether memory optimization is enabled.
 *
 * <p>However, unlike {@link KNNEngine},
 * the input here is a Faiss score, which must be converted to Lucene’s
 * scoring range.</p>
 *
 * <p>For example, Faiss uses inner product while Lucene uses
 * maximum inner product. When converting distances, this class maps
 * the Faiss score into the maximum inner product range so Lucene can
 * interpret it correctly during search.</p>
 *
 * <p>Conversely, it also converts Lucene scores back into Faiss scores so that
 * the same query produces consistent results across both implementations.
 *
 * <p>Note that this should be used only when memory_optimized_search is enabled.
 *
 */
public final class MemoryOptimizedSearchScoreConverter {
    /**
     * Convert Faiss distance to Lucene score.
     *
     * @param distance Faiss distance
     * @param spaceType Space type being used.
     * @return Converted value to be used during Lucene search algorithm.
     */
    public static float distanceToRadialThreshold(final float distance, final SpaceType spaceType) {
        if (spaceType != SpaceType.COSINESIMIL) {
            return KNNEngine.LUCENE.distanceToRadialThreshold(distance, spaceType);
        }

        // For cosine similarity, `distance = 1 - inner_product_value`.
        // therefore, we should extract it then convert it to max_inner_product_value
        final float innerProductValue = KNNEngine.FAISS.distanceToRadialThreshold(distance, SpaceType.COSINESIMIL);

        // Convert inner product value to max inner product value.
        return SpaceType.INNER_PRODUCT.scoreTranslation(-innerProductValue);
    }

    /**
     * Convert Faiss score to Lucene radial threshold.
     *
     * @param score Faiss score
     * @param spaceType Space type that's being used
     * @return Converted radial threshold for Lucene
     */
    public static float scoreToRadialThreshold(final float score, final SpaceType spaceType) {
        if (spaceType != SpaceType.COSINESIMIL) {
            return KNNEngine.LUCENE.scoreToRadialThreshold(score, spaceType);
        }

        // Since `score = (2 - (1 - inner_product_value)) / 2 = (1 + inner_product_value) / 2`,
        // we should extract it then convert it to max inner product value.
        final float innerProductValue = KNNEngine.FAISS.scoreToRadialThreshold(score, SpaceType.COSINESIMIL);

        // Convert inner product value to max inner product value.
        return SpaceType.INNER_PRODUCT.scoreTranslation(-innerProductValue);
    }

    /**
     * This method converts Lucene's max inner product score to Faiss cosine score to ensure user
     * to get the same results with the same query.
     *
     * @param scoreDocs Results from internal search before returning.
     */
    public static void convertToCosineScore(final ScoreDoc[] scoreDocs) {
        for (final ScoreDoc scoreDoc : scoreDocs) {
            // For cosine similarity, MAXIMUM_INNER_PRODUCT being used internally.
            // Which maps negative values (which is plain inner product result value) to (0, 1], and maps positive values to (1, +inf).
            // Below logic is to reverse back and extract the result of plain inner product.
            final float innerProductValue;
            if (scoreDoc.score >= 1) {
                // Inner product value is positive.
                innerProductValue = scoreDoc.score - 1;
            } else {
                // Inner product value is negative.
                innerProductValue = 1 - 1 / scoreDoc.score;
            }

            // Then we need to transform the value to be bounded the desired range in cosine similarity space type.
            scoreDoc.score = KNNEngine.FAISS.score(innerProductValue, SpaceType.COSINESIMIL);
        }
    }
}
