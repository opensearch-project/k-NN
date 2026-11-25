/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.opensearch.knn.KNNTestCase;

import java.util.Locale;
import java.util.Map;

public class MMRExplainInfoTests extends KNNTestCase {

    public void testToMap_withFirstSelectedDocument() {
        MMRExplainInfo explainInfo = MMRExplainInfo.builder()
            .originalScore(0.9f)
            .maxSimilarityToSelected(0.0f)
            .mmrScore(0.45)
            .diversity(0.5f)
            .build();

        Map<String, Object> map = explainInfo.toMap();

        assertEquals(0.9f, map.get(MMRExplainInfo.ORIGINAL_SCORE_FIELD));
        assertEquals(0.0f, map.get(MMRExplainInfo.MAX_SIMILARITY_TO_SELECTED_FIELD));
        assertEquals(0.45, map.get(MMRExplainInfo.MMR_SCORE_FIELD));
        assertTrue(map.get(MMRExplainInfo.MMR_FORMULA_FIELD).toString().contains("(1 - 0.5000) * 0.9000 - 0.5000 * 0.0000 = 0.4500"));
    }

    public void testToMap_withSubsequentSelectedDocument() {
        MMRExplainInfo explainInfo = MMRExplainInfo.builder()
            .originalScore(0.8f)
            .maxSimilarityToSelected(0.6f)
            .mmrScore(0.1)
            .diversity(0.5f)
            .build();

        Map<String, Object> map = explainInfo.toMap();

        assertEquals(0.8f, map.get(MMRExplainInfo.ORIGINAL_SCORE_FIELD));
        assertEquals(0.6f, map.get(MMRExplainInfo.MAX_SIMILARITY_TO_SELECTED_FIELD));
        assertEquals(0.1, map.get(MMRExplainInfo.MMR_SCORE_FIELD));
        String formula = map.get(MMRExplainInfo.MMR_FORMULA_FIELD).toString();
        assertEquals(String.format(Locale.ROOT, "(1 - %.4f) * %.4f - %.4f * %.4f = %.4f", 0.5f, 0.8f, 0.5f, 0.6f, 0.1), formula);
    }

    public void testToMap_preservesFieldOrder() {
        MMRExplainInfo explainInfo = MMRExplainInfo.builder()
            .originalScore(1.0f)
            .maxSimilarityToSelected(0.5f)
            .mmrScore(0.25)
            .diversity(0.5f)
            .build();

        Map<String, Object> map = explainInfo.toMap();
        String[] keys = map.keySet().toArray(new String[0]);

        assertEquals(MMRExplainInfo.ORIGINAL_SCORE_FIELD, keys[0]);
        assertEquals(MMRExplainInfo.MAX_SIMILARITY_TO_SELECTED_FIELD, keys[1]);
        assertEquals(MMRExplainInfo.MMR_SCORE_FIELD, keys[2]);
        assertEquals(MMRExplainInfo.MMR_FORMULA_FIELD, keys[3]);
    }

    public void testToMap_withZeroDiversity() {
        MMRExplainInfo explainInfo = MMRExplainInfo.builder()
            .originalScore(0.75f)
            .maxSimilarityToSelected(0.3f)
            .mmrScore(0.75)
            .diversity(0.0f)
            .build();

        Map<String, Object> map = explainInfo.toMap();

        assertEquals(0.75f, map.get(MMRExplainInfo.ORIGINAL_SCORE_FIELD));
        assertTrue(map.get(MMRExplainInfo.MMR_FORMULA_FIELD).toString().contains("0.0000"));
    }
}
