/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import lombok.Builder;
import lombok.Value;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * A DTO to hold the explain info for a single hit selected by the MMR algorithm.
 * This captures the scoring details at the moment the document was chosen.
 * Note: the selection order and previously-selected documents can be inferred from
 * the position of each hit in the result list.
 */
@Value
@Builder
public class MMRExplainInfo {

    public static final String ORIGINAL_SCORE_FIELD = "original_score";
    public static final String MAX_SIMILARITY_TO_SELECTED_FIELD = "max_similarity_to_selected";
    public static final String MMR_SCORE_FIELD = "mmr_score";
    public static final String MMR_FORMULA_FIELD = "mmr_formula";

    /**
     * The original relevance score from the KNN/neural search.
     */
    float originalScore;

    /**
     * The maximum vector similarity between this document and any already-selected document.
     * For the first selected document, this is 0.0.
     */
    float maxSimilarityToSelected;

    /**
     * The computed MMR score at selection time: (1 - diversity) * originalScore - diversity * maxSimilarityToSelected.
     */
    double mmrScore;

    /**
     * The diversity parameter (lambda) used in the MMR formula.
     */
    float diversity;

    /**
     * Converts this explain info to a map suitable for injection into a search hit's _source.
     *
     * @return a map representing the mmr_explain payload
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put(ORIGINAL_SCORE_FIELD, originalScore);
        map.put(MAX_SIMILARITY_TO_SELECTED_FIELD, maxSimilarityToSelected);
        map.put(MMR_SCORE_FIELD, mmrScore);
        map.put(
            MMR_FORMULA_FIELD,
            String.format(
                Locale.ROOT,
                "(1 - %.4f) * %.4f - %.4f * %.4f = %.4f",
                diversity,
                originalScore,
                diversity,
                maxSimilarityToSelected,
                mmrScore
            )
        );
        return map;
    }
}
