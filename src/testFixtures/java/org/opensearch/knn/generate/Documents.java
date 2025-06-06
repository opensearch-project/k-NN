/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.generate;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.VectorDataType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.opensearch.knn.generate.DocumentsGenerator.FILTER_ID_NO_MOD;

/**
 * This represents a set of document for one batch.
 * For the structure of generated document, refer to comment on {@link DocumentsGenerator}.
 * User can generate this first by using {@link DocumentsGenerator}, then set prepareAnswerSet with a query vector.
 * After prepared answer set, user can proceed to validate result with validateResponse method.
 */
@RequiredArgsConstructor
public class Documents {
    public record Result(String id, String filterId) {
    }

    // Document JSON strings. documents[i] indicates ith document JSON string.
    @Getter
    private final List<String> documents;
    // Vectors generated per each document. vectors[i] indicates vectors in ith document.
    // Note that non-nested case, length of vectors[i] will be 1. Otherwise, it will have 3 vectors meaning one nested field
    // having three nested child documents.
    private final List<List<float[]>> vectors;
    // Expected document ids to be set in prepareAnswerSet against to those generated documents.
    private Set<String> expectedIds;
    // Whether we need filtering or no.
    private boolean doFiltering;

    public float prepareAnswerSet(
        final Object queryVector,
        final KNNVectorSimilarityFunction similarityFunction,
        final boolean doFiltering,
        final boolean isRadial
    ) {

        // Save whether we applied filtering.
        this.doFiltering = doFiltering;

        assert documents.size() == vectors.size();

        final Map<String, Float> scoreTable = new HashMap<>();
        final List<Float> similarities = new ArrayList<>();

        for (int i = 0; i < vectors.size(); i++) {
            // This doc does not have KNN field.
            if (vectors.get(i).isEmpty()) {
                continue;
            }

            // If filtering exists, it only docs having 'filter-0'
            if (doFiltering && (i % FILTER_ID_NO_MOD) == 1) {
                continue;
            }

            // Find the best child
            float bestSimilarity = Float.MIN_VALUE;
            for (float[] vector : vectors.get(i)) {
                final float score;
                if (queryVector instanceof float[]) {
                    score = similarityFunction.compare((float[]) queryVector, vector);
                } else if (queryVector instanceof byte[]) {
                    score = similarityFunction.compare((byte[]) queryVector, SearchTestHelper.convertToByteArray(vector));
                } else {
                    throw new AssertionError();
                }
                bestSimilarity = Math.max(bestSimilarity, score);
            }
            similarities.add(bestSimilarity);
            scoreTable.put("id-" + i, bestSimilarity);
        }

        // Recommend minimum similarity for radial search
        float minSimilarity = Float.MIN_VALUE;
        if (isRadial) {
            // Pick median similarity
            similarities.sort(Collections.reverseOrder());
            minSimilarity = similarities.get(similarities.size() / 2);

            final Map<String, Float> newScoreTable = new HashMap<>();
            for (final Map.Entry<String, Float> entry : scoreTable.entrySet()) {
                if (entry.getValue() >= minSimilarity) {
                    newScoreTable.put(entry.getKey(), entry.getValue());
                }
            }

            scoreTable.clear();
            scoreTable.putAll(newScoreTable);
        }

        // Sort the results
        final List<String> ids = new ArrayList<>(scoreTable.keySet());
        Collections.sort(ids, (id0, id1) -> -Float.compare(scoreTable.get(id0), scoreTable.get(id1)));

        // Save the answer
        expectedIds = new HashSet<>(ids);

        return minSimilarity;
    }

    public float prepareAnswerSet(
        final VectorDataType dataType,
        final byte[] vector,
        final KNNVectorSimilarityFunction similarityFunction,
        final boolean doFiltering,
        final boolean isRadial
    ) {
        assert (dataType == VectorDataType.BYTE || dataType == VectorDataType.BINARY);
        return prepareAnswerSet(vector, similarityFunction, doFiltering, isRadial);
    }

    public void validateResponse(final List<Result> results) {
        // Filtering check
        if (doFiltering) {
            for (final Result result : results) {
                assertEquals(result.filterId, "filter-0");
            }
        }

        // Results validation.
        int matchCount = 0;
        for (final Result result : results) {
            if (expectedIds.contains(result.id)) {
                ++matchCount;
            }
        }

        // At least we must have 0.8 recall.
        float matchRatio = (float) matchCount / (float) results.size();
        assertTrue("Match ratio[=" + matchRatio + "] <= 0.8", matchRatio > 0.8);
    }
}
