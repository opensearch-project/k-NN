/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.util.List;
import java.util.Map;

import org.opensearch.knn.index.SpaceType;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.NonNull;

/**
 * Utility class providing k-NN scoring functions for Painless scripts.
 * This class implements late interaction scoring for multi-vector similarity search,
 * commonly used in ColBERT-style retrieval systems.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class KNNPainlessScriptUtils {

    /**
     * Calculates the late interaction score between query vectors and document vectors.
     * For each query vector, finds the maximum similarity with any document vector and sums these maxima.
     * This implements a ColBERT-style late interaction pattern for token-level matching.
     *
     * @param queryVectors List of query vectors, each a float array
     * @param docFieldName Name of the field in the document containing vectors
     * @param doc Document source as a map
     * @return Sum of maximum similarity scores
     */
    @SuppressWarnings("unchecked")
    public static double lateInteractionScore(
        @NonNull final List<float[]> queryVectors,
        @NonNull final String docFieldName,
        @NonNull final Map<String, Object> doc
    ) {
        return lateInteractionScore(queryVectors, docFieldName, doc, "innerproduct");
    }

    /**
     * Calculates the late interaction score between query vectors and document vectors with specified similarity metric.
     * For each query vector, finds the maximum similarity with any document vector and sums these maxima.
     * This implements a ColBERT-style late interaction pattern for token-level matching.
     *
     * @param queryVectors List of query vectors, each a float array
     * @param docFieldName Name of the field in the document containing vectors
     * @param doc Document source as a map
     * @param spaceType Space type for similarity calculation: "innerproduct", "cosinesimil", "l2", "l1", "linf"
     * @return Sum of maximum similarity scores
     */
    @SuppressWarnings("unchecked")
    public static double lateInteractionScore(
        @NonNull final List<float[]> queryVectors,
        @NonNull final String docFieldName,
        @NonNull final Map<String, Object> doc,
        @NonNull final String spaceType
    ) {
        if (queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Query vectors cannot be empty");
        }

        List<float[]> docVectors = (List<float[]>) doc.get(docFieldName);
        if (docVectors == null || docVectors.isEmpty()) {
            return 0.0;
        }

        SpaceType space = SpaceType.getSpace(spaceType);
        boolean isDistanceMetric = space == SpaceType.L2 || space == SpaceType.L1 || space == SpaceType.LINF;

        double totalMaxSim = 0.0;
        int expectedDimension = -1;

        for (float[] q_vec : queryVectors) {
            if (q_vec == null || q_vec.length == 0) {
                throw new IllegalArgumentException("Every single vector within query vectors cannot be empty or null");
            }

            // Set expected dimension from first valid query vector
            if (expectedDimension == -1) {
                expectedDimension = q_vec.length;
            } else if (q_vec.length != expectedDimension) {
                throw new IllegalArgumentException(
                    String.format("Query vector dimension mismatch: expected %d, found %d", expectedDimension, q_vec.length)
                );
            }

            double maxDocTokenSim = isDistanceMetric ? Double.MAX_VALUE : Double.NEGATIVE_INFINITY;

            for (float[] doc_token_vec : docVectors) {
                if (doc_token_vec == null || doc_token_vec.length == 0) {
                    continue;
                }

                // Validate doc vector dimension
                if (doc_token_vec.length != expectedDimension) {
                    throw new IllegalArgumentException(
                        String.format("Document vector dimension mismatch: expected %d, found %d", expectedDimension, doc_token_vec.length)
                    );
                }

                double currentSim = calculateSimilarity(q_vec, doc_token_vec, space);

                if (isDistanceMetric) {
                    // For distance metrics (L2, L1, LINF), we want minimum distance
                    if (currentSim < maxDocTokenSim) {
                        maxDocTokenSim = currentSim;
                    }
                } else {
                    // For similarity metrics (inner product, cosine), we want maximum similarity
                    if (currentSim > maxDocTokenSim) {
                        maxDocTokenSim = currentSim;
                    }
                }
            }

            // Convert distance to similarity score using SpaceType's scoreTranslation
            if (isDistanceMetric) {
                maxDocTokenSim = space.scoreTranslation((float) maxDocTokenSim);
            }

            totalMaxSim += maxDocTokenSim;
        }
        return totalMaxSim;
    }

    private static double calculateSimilarity(
        @NonNull final float[] vec1,
        @NonNull final float[] vec2,
        @NonNull final SpaceType spaceType
    ) {
        // Note: Dimension validation is now done in the main loop for efficiency
        switch (spaceType) {
            case INNER_PRODUCT:
                return KNNScoringUtil.innerProduct(vec1, vec2);
            case COSINESIMIL:
                return KNNScoringUtil.cosinesimil(vec1, vec2);
            case L2:
                return KNNScoringUtil.l2Squared(vec1, vec2);
            case L1:
                return KNNScoringUtil.l1Norm(vec1, vec2);
            case LINF:
                return KNNScoringUtil.lInfNorm(vec1, vec2);
            default:
                return KNNScoringUtil.innerProduct(vec1, vec2);
        }
    }
}
