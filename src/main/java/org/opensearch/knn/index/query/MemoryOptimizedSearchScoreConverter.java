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
        if (spaceType == SpaceType.INNER_PRODUCT) {
            // Faiss distance for IP is -dot. Negate to recover the raw dot product before Lucene's conversion.
            return KNNEngine.LUCENE.distanceToRadialThreshold(-distance, spaceType);
        }
        // The memory-optimized scorer emits Lucene-native scores for cosine and L2, including
        // cosine (FP16_COSINE / SQ_COSINE apply the (1 + dot) / 2 transform in-kernel). Therefore the
        // radial threshold is expressed in Lucene score space, so we can delegate to the Lucene engine
        // directly. For COSINESIMIL this yields (2 - distance) / 2, matching the scorer output.
        //
        // The ADC (binary-quantized) path is the one exception: it scores a float query against 1-bit
        // vectors and emits MaxIP-format scores that MemoryOptimizedKNNWeight post-converts to cosine.
        // Those scores are NOT Lucene-native, so this converter would be wrong for them. That mismatch is
        // unreachable, however, because ADC requires the BQ encoder and ResolvedIndexSpec#supportsRadialSearch()
        // rejects radial search for BQ (and every quantized index) up front in KNNQueryBuilder. If radial
        // search is ever enabled for BQ/ADC, this method (and scoreToRadialThreshold below) must convert the
        // threshold from cosine to MaxIP space for the ADC case first.
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
        // The memory-optimized scorer emits Lucene-native scores for every space type reachable here,
        // including cosine (FP16_COSINE / SQ_COSINE apply the (1 + dot) / 2 transform in-kernel). The
        // user-supplied min_score is already in that same Lucene score space, so we can delegate to the
        // Lucene engine uniformly, which returns the score unchanged.
        //
        // The ADC (binary-quantized) cosine path emits MaxIP-format scores (post-converted to cosine in
        // MemoryOptimizedKNNWeight) rather than Lucene-native scores, so it would need the threshold
        // converted from cosine to MaxIP space. It never reaches this method: ADC requires the BQ encoder,
        // and ResolvedIndexSpec#supportsRadialSearch() rejects radial search for BQ up front. If that gate
        // is ever relaxed for BQ/ADC, add the ADC threshold conversion here (see distanceToRadialThreshold).
        return KNNEngine.LUCENE.scoreToRadialThreshold(score, spaceType);
    }

    /**
     * This method converts Lucene's max inner product score to Faiss cosine score to ensure user
     * to get the same results with the same query.
     *
     * @param scoreDocs Results from internal search before returning.
     */
    public static void convertToCosineScore(final ScoreDoc[] scoreDocs) {
        for (final ScoreDoc scoreDoc : scoreDocs) {
            scoreDoc.score = convertInnerProductScoreToCosineScore(scoreDoc.score);
        }
    }

    /**
     * Converts a single Lucene MAXIMUM_INNER_PRODUCT score to a Faiss cosine similarity score.
     *
     * <p>MAXIMUM_INNER_PRODUCT maps negative inner product values to (0, 1] and positive values
     * to (1, +inf). This method reverses that mapping to recover the raw inner product value,
     * then transforms it into the cosine similarity score range.</p>
     *
     * @param ipScore the MAXIMUM_INNER_PRODUCT-format score
     * @return the equivalent cosine similarity score
     */
    public static float convertInnerProductScoreToCosineScore(final float ipScore) {
        // Reverse MAXIMUM_INNER_PRODUCT score translation to recover the raw inner product value.
        final float innerProductValue = ipScore >= 1 ? ipScore - 1 : 1 - 1 / ipScore;
        // Transform to cosine similarity score range.
        return KNNEngine.FAISS.score(innerProductValue, SpaceType.COSINESIMIL);
    }
}
