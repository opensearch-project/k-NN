/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.util.VectorUtil;
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
     * See {@link #lateInteractionScore(List, String, Map, String)} for the score definition.
     *
     * @param queryVectors List of query vectors
     * @param docFieldName Name of the field in the document containing vectors
     * @param doc Document source as a map
     * @return Non-negative score that is monotone in the MaxSim of the query and document vectors
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
     * This implements ColBERT-style MaxSim: for each query vector, take the best raw similarity with any document
     * vector, and sum these over the query vectors. The raw similarity is the dot product for "innerproduct", the
     * cosine for "cosinesimil", and the negated squared Euclidean distance for "l2", so "l2" ranks documents by the
     * smallest sum of per-query-vector minimum squared distances.
     * <p>
     * Because script_score rejects negative scores, the MaxSim sum S over n query vectors is returned as
     * {@code n * f(S / n)}, where f is the space's usual k-NN score transform: {@code x + 1} for x &gt;= 0 and
     * {@code 1 / (1 - x)} for x &lt; 0 with "innerproduct", {@code (1 + x) / 2} with "cosinesimil", and
     * {@code 1 / (1 + d)} of the mean minimum squared distance d with "l2". The score is strictly monotone in
     * MaxSim, so documents are ranked in MaxSim order.
     *
     * @param queryVectors List of query vectors
     * @param docFieldName Name of the field in the document containing vectors
     * @param doc Document source as a map
     * @param spaceType Space type for similarity calculation: "innerproduct", "cosinesimil" or "l2".
     *                  Other space types, including "l1" and "linf", throw {@link IllegalArgumentException}.
     * @return Non-negative score that is monotone in the MaxSim of the query and document vectors
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
        if (space != SpaceType.INNER_PRODUCT && space != SpaceType.COSINESIMIL && space != SpaceType.L2) {
            throw new IllegalArgumentException("Space type " + spaceType + " is not supported for late interaction scoring");
        }
        final boolean isL2 = space == SpaceType.L2;

        // Convert document vectors to primitive arrays once, outside the query-vector loop.
        // Previously this conversion ran once per (query vector, document vector) pair, so a
        // document with D vectors scored against Q query vectors paid Q*D float[] allocations
        // and Q*D*dim Number unboxing calls, which dominates latency for large multi-vector
        // fields (e.g. ~1M allocations per scored document at Q=D=1024).
        float[][] convertedDocVectors = new float[docVectors.size()][];
        int numValidDocVectors = 0;
        for (List<Number> docVector : docVectors) {
            if (docVector == null || docVector.isEmpty()) {
                continue;
            }
            float[] dVec = new float[docVector.size()];
            for (int i = 0; i < docVector.size(); i++) {
                dVec[i] = docVector.get(i).floatValue();
            }
            convertedDocVectors[numValidDocVectors++] = dVec;
        }

        // Sum the raw per-token similarities (dot product, cosine, or minimum squared L2 distance). Summing
        // Lucene's per-token scaled scores instead does not preserve MaxSim ordering, because the l2 and
        // innerproduct scalings are not affine.
        double rawSum = 0.0;
        int numScoredQueryVectors = 0;

        for (List<Number> queryVector : queryVectors) {
            if (queryVector == null || queryVector.isEmpty()) {
                throw new IllegalArgumentException("Every single vector within query vectors cannot be empty or null");
            }

            float[] qVec = new float[queryVector.size()];
            for (int i = 0; i < queryVector.size(); i++) {
                qVec[i] = queryVector.get(i).floatValue();
            }

            float best = isL2 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            for (int i = 0; i < numValidDocVectors; i++) {
                float[] dVec = convertedDocVectors[i];
                switch (space) {
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

            if (numValidDocVectors > 0) {
                rawSum += best;
                numScoredQueryVectors++;
            }
        }

        if (numScoredQueryVectors == 0) {
            return 0.0f;
        }
        return toScore(space, rawSum, numScoredQueryVectors);
    }

    /**
     * Maps the raw MaxSim sum to a non-negative score, since script_score rejects negative scores. The space's
     * Lucene score transform is applied once, to the per-query-token mean, and the result is multiplied by the
     * number of query tokens. The mapping is strictly monotone in the raw sum, so ranking follows MaxSim, and it
     * reproduces the per-token-summed score wherever that was already order-preserving: always for cosinesimil, for
     * innerproduct when no per-token maximum is negative, and for every space when there is one query token.
     */
    private static float toScore(final SpaceType space, final double rawSum, final int numQueryVectors) {
        final double mean = rawSum / numQueryVectors;
        switch (space) {
            case L2:
                return (float) (numQueryVectors / (1 + mean));
            case COSINESIMIL:
                return (float) (numQueryVectors * (1 + mean) / 2);
            default:
                return (float) (numQueryVectors * (mean < 0 ? 1 / (1 - mean) : mean + 1));
        }
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
