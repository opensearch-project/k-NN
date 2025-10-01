/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.core.common.Strings;

import java.util.List;
import java.util.Map;

/**
 * Utility class for k-NN scoring functions used in Painless scripts.
 * Provides late interaction scoring functionality for ColBERT-style token-level matching.
 */
public class KNNPainlessScriptUtils {

    /**
     * Calculates the late interaction score between query vectors and document vectors using default similarity metric.
     * The default similarity metric is determined by the default space type, which is L2.
     * For each query vector, finds the maximum similarity with any document vector and sums these maxima.
     * This implements a ColBERT-style late interaction pattern for token-level matching.
     *
     * @param queryVectors List of query vectors
     * @param docFieldName Name of the field in the document containing vectors
     * @param doc Document source as a map
     * @return Sum of maximum similarity scores
     */
    public static float lateInteractionScore(
        final List<List<Number>> queryVectors,
        final String docFieldName,
        final Map<String, Object> doc
    ) {
        return lateInteractionScore(queryVectors, docFieldName, doc, SpaceType.DEFAULT.getValue());
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
        final List<List<Number>> queryVectors,
        final String docFieldName,
        final Map<String, Object> doc,
        final String spaceType
    ) {
        validateInputs(queryVectors, docFieldName, doc, spaceType);

        List<List<Number>> docVectors;
        try {
            docVectors = (List<List<Number>>) doc.get(docFieldName);
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("Field " + docFieldName + " must contain a list of vector lists", e);
        }

        if (docVectors == null || docVectors.isEmpty()) {
            throw new IllegalArgumentException("Document vectors cannot be null or empty");
        }

        SpaceType space = SpaceType.getSpace(spaceType);
        KNNVectorSimilarityFunction similarityFunction = space.getKnnVectorSimilarityFunction();
        if (similarityFunction == null) {
            throw new IllegalArgumentException("Space type " + spaceType + " does not support vector similarity function");
        }

        float totalScore = 0.0f;

        for (List<Number> queryVector : queryVectors) {
            if (queryVector == null || queryVector.isEmpty()) {
                throw new IllegalArgumentException("Every single vector within query vectors cannot be empty or null");
            }

            float[] qVec = new float[queryVector.size()];
            for (int i = 0; i < queryVector.size(); i++) {
                qVec[i] = queryVector.get(i).floatValue();
            }

            float bestRawScore = (float) docVectors.parallelStream()
                .filter(docVector -> docVector != null && !docVector.isEmpty())
                .mapToDouble(docVector -> {
                    float[] dVec = new float[docVector.size()];
                    for (int i = 0; i < docVector.size(); i++) {
                        dVec[i] = docVector.get(i).floatValue();
                    }
                    return similarityFunction.compare(qVec, dVec);
                })
                .max()
                .orElse(Double.NEGATIVE_INFINITY);

            if (bestRawScore != Double.NEGATIVE_INFINITY) {
                totalScore += bestRawScore;
            }
        }
        return totalScore;
    }

    private static void validateInputs(
        final List<List<Number>> queryVectors,
        final String docFieldName,
        final Map<String, Object> doc,
        final String spaceType
    ) {
        if (queryVectors == null) {
            throw new IllegalArgumentException("Query vectors cannot be null");
        }
        if (queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Query vectors cannot be empty");
        }
        if (Strings.isNullOrEmpty(docFieldName)) {
            throw new IllegalArgumentException("Document field name cannot be null or empty");
        }
        if (doc == null) {
            throw new IllegalArgumentException("Document cannot be null");
        }
        if (Strings.isNullOrEmpty(spaceType)) {
            throw new IllegalArgumentException("Space type cannot be null or empty");
        }
    }
}
