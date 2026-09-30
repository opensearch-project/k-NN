/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.MultiVectorSimilarity;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.index.SpaceType;

/**
 * ColBERT-style MaxSim over <b>raw</b> per-token similarities, for the native late-interaction rescore.
 *
 * <p>Lucene's built-in {@code LateInteractionFloatValuesSource.ScoreFunction.SUM_MAX_SIM} computes MaxSim
 * by summing {@link VectorSimilarityFunction#compare} per query token, i.e. Lucene's <em>scaled</em>
 * per-token scores ({@code 1/(1+d²)} for euclidean, the piecewise {@code x+1} / {@code 1/(1-x)} for
 * maximum-inner-product). A sum of per-token monotone transforms only preserves MaxSim order when the
 * transform is affine, so that scaling can rank documents out of MaxSim order for {@code l2}, and for
 * {@code innerproduct} when a query token's best dot product is negative (see k-NN #3612).
 *
 * <p>This implementation instead sums the raw per-token similarity — dot product for
 * {@link SpaceType#INNER_PRODUCT}, cosine for {@link SpaceType#COSINESIMIL}, and the minimum squared
 * Euclidean distance for {@link SpaceType#L2} — and applies the space's score transform once, to the
 * per-query-token mean. The result is non-negative and strictly monotone in MaxSim, so documents rank in
 * MaxSim order. This mirrors the Painless {@code lateInteractionScore} fix in #3613 so the two paths agree.
 */
public final class RawMaxSimilarity implements MultiVectorSimilarity {

    private final SpaceType spaceType;

    public RawMaxSimilarity(final SpaceType spaceType) {
        if (spaceType != SpaceType.INNER_PRODUCT && spaceType != SpaceType.COSINESIMIL && spaceType != SpaceType.L2) {
            throw new IllegalArgumentException("Space type " + spaceType.getValue() + " is not supported for late interaction scoring");
        }
        this.spaceType = spaceType;
    }

    /**
     * @param queryVector              per-token query multi-vectors
     * @param docVector                per-token document multi-vectors
     * @param vectorSimilarityFunction unused; the raw similarity is selected from the space type instead of
     *                                 Lucene's scaled {@code compare}. Present to satisfy the interface.
     * @return non-negative score, strictly monotone in the MaxSim of the query and document token sets
     */
    @Override
    public float compare(final float[][] queryVector, final float[][] docVector, final VectorSimilarityFunction vectorSimilarityFunction) {
        if (queryVector.length == 0 || docVector.length == 0) {
            return 0.0f;
        }
        final boolean isL2 = spaceType == SpaceType.L2;
        double rawSum = 0.0;
        for (final float[] qVec : queryVector) {
            float best = isL2 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            for (final float[] dVec : docVector) {
                switch (spaceType) {
                    case L2:
                        best = Math.min(best, VectorUtil.squareDistance(qVec, dVec));
                        break;
                    case COSINESIMIL:
                        best = Math.max(best, VectorUtil.cosine(qVec, dVec));
                        break;
                    default:
                        best = Math.max(best, VectorUtil.dotProduct(qVec, dVec));
                }
            }
            rawSum += best;
        }
        return toScore(rawSum, queryVector.length);
    }

    /**
     * Maps the raw MaxSim sum to a non-negative score, applying the space's score transform once to the
     * per-query-token mean and scaling back by the token count. Strictly monotone in the raw sum, so ranking
     * follows MaxSim. Matches {@code KNNPainlessScriptUtils#lateInteractionScore} (#3613) exactly.
     */
    private float toScore(final double rawSum, final int numQueryVectors) {
        final double mean = rawSum / numQueryVectors;
        switch (spaceType) {
            case L2:
                return (float) (numQueryVectors / (1 + mean));
            case COSINESIMIL:
                return (float) (numQueryVectors * (1 + mean) / 2);
            default:
                return (float) (numQueryVectors * (mean < 0 ? 1 / (1 - mean) : mean + 1));
        }
    }
}
