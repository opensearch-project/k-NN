/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import lombok.NonNull;
import org.opensearch.knn.index.SpaceType;

import java.util.List;
import java.util.Map;

/**
 * Utility class for k-NN scoring functions used in Painless scripts.
 * Provides late interaction scoring functionality for ColBERT-style token-level matching.
 */
public class KNNPainlessScriptUtils {

    /**
     * Calculates the late interaction score between query vectors and document vectors using inner product.
     * For each query vector, finds the maximum similarity with any document vector and sums these maxima.
     * This implements a ColBERT-style late interaction pattern for token-level matching.
     *
     * @param queryVectors List of query vectors
     * @param docFieldName Name of the field in the document containing vectors
     * @param doc Document source as a map
     * @return Sum of maximum similarity scores
     */
    public static float lateInteractionScore(
        @NonNull final List<List<Number>> queryVectors,
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
     * @param queryVectors List of query vectors
     * @param docFieldName Name of the field in the document containing vectors
     * @param doc Document source as a map
     * @param spaceType Space type for similarity calculation: "innerproduct", "cosinesimil", "l2", "l1", "linf"
     * @return Sum of maximum similarity scores
     */
    @SuppressWarnings("unchecked")
    public static float lateInteractionScore(
        @NonNull final List<List<Number>> queryVectors,
        @NonNull final String docFieldName,
        @NonNull final Map<String, Object> doc,
        @NonNull final String spaceType
    ) {
        List<List<Number>> docVectors = (List<List<Number>>) doc.get(docFieldName);

        if (queryVectors.isEmpty() || docVectors == null || docVectors.isEmpty()) {
            return 0.0f;
        }

        float totalMaxSim = 0.0f;
        SpaceType space = SpaceType.getSpace(spaceType);
        boolean isDistanceMetric = space == SpaceType.L2 || space == SpaceType.L1 || space == SpaceType.LINF;

        for (List<Number> queryVector : queryVectors) {
            if (queryVector == null || queryVector.isEmpty()) {
                throw new IllegalArgumentException("Every single vector within query vectors cannot be empty or null");
            }

            float[] qVec = new float[queryVector.size()];
            for (int i = 0; i < queryVector.size(); i++) {
                qVec[i] = queryVector.get(i).floatValue();
            }

            float maxDocTokenSim = isDistanceMetric ? Float.MAX_VALUE : Float.MIN_VALUE;

            for (List<Number> docVector : docVectors) {
                if (docVector == null || docVector.isEmpty()) {
                    continue;
                }

                float[] dVec = new float[docVector.size()];
                for (int i = 0; i < docVector.size(); i++) {
                    dVec[i] = docVector.get(i).floatValue();
                }

                float similarity = calculateSimilarity(qVec, dVec, space);

                if (isDistanceMetric) {
                    maxDocTokenSim = Math.min(maxDocTokenSim, similarity);
                } else {
                    maxDocTokenSim = Math.max(maxDocTokenSim, similarity);
                }
            }

            totalMaxSim += maxDocTokenSim;
        }
        return totalMaxSim;
    }

    private static float calculateSimilarity(@NonNull final float[] vec1, @NonNull final float[] vec2, @NonNull final SpaceType spaceType) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException(
                String.format(
                    "Vector dimension mismatch in lateInteractionScore: query vector has %d dimensions, document vector has %d dimensions. "
                        + "Ensure all vectors use the same dimensionality from the same embedding model.",
                    vec1.length,
                    vec2.length
                )
            );
        }

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
