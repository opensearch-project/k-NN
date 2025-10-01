/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.test.OpenSearchTestCase;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Unit tests for KNNPainlessScriptUtils class.
 * Tests late interaction scoring functionality with various input scenarios.
 */
public class KNNPainlessScriptUtilsTests extends OpenSearchTestCase {

    /**
     * Tests late interaction score calculation with valid input vectors using inner product.
     */
    public void testLateInteractionScore_whenValidVectors_thenReturnsCorrectScore() {
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv1 = new ArrayList<>();
        qv1.add(1.0);
        qv1.add(0.0);
        queryVectors.add(qv1);

        List<List<Number>> docVectors = new ArrayList<>();
        List<Number> dv1 = new ArrayList<>();
        dv1.add(1.0);
        dv1.add(0.0);
        docVectors.add(dv1);

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        float actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "innerproduct");
        assertTrue("Score should be positive", actual > 0);
    }

    /**
     * Tests late interaction score with empty vectors.
     */
    public void testLateInteractionScore_whenEmptyVectors_thenThrowsException() {
        List<List<Number>> emptyQueryVectors = new ArrayList<>();
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", new ArrayList<List<Number>>());

        // Empty query vectors should throw exception
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(emptyQueryVectors, "my_vector", doc)
        );

        // Empty document vectors should throw exception
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv = new ArrayList<>();
        qv.add(1.0);
        queryVectors.add(qv);

        expectThrows(IllegalArgumentException.class, () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc));

        // Null document vectors should throw exception
        Map<String, Object> docWithNull = new HashMap<>();
        docWithNull.put("my_vector", null);

        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", docWithNull)
        );
    }

    /**
     * Tests late interaction score with null inputs.
     */
    public void testLateInteractionScore_whenNullInputs_thenThrowsException() {
        List<List<Number>> queryVectors = new ArrayList<>();
        Map<String, Object> doc = new HashMap<>();

        // Test null query vectors
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(null, "my_vector", doc, "innerproduct")
        );

        // Test null field name
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, null, doc, "innerproduct")
        );

        // Test null document
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", null, "innerproduct")
        );

        // Test null space type
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, null)
        );
    }

    /**
     * Tests late interaction score with invalid field type.
     */
    public void testLateInteractionScore_whenInvalidFieldType_thenThrowsException() {
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv = new ArrayList<>();
        qv.add(1.0);
        queryVectors.add(qv);

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", "invalid_type"); // String instead of List<List<Number>>

        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "innerproduct")
        );
    }

    /**
     * Tests late interaction score with unsupported space type.
     */
    public void testLateInteractionScore_whenUnsupportedSpaceType_thenThrowsException() {
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv = new ArrayList<>();
        qv.add(1.0);
        qv.add(0.0);
        queryVectors.add(qv);

        List<List<Number>> docVectors = new ArrayList<>();
        List<Number> dv = new ArrayList<>();
        dv.add(1.0);
        dv.add(0.0);
        docVectors.add(dv);

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // L1 and LINF don't have KNNVectorSimilarityFunction support
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "l1")
        );

        expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "linf")
        );
    }

    /**
     * Tests late interaction score with different supported space types.
     */
    public void testLateInteractionScore_whenSupportedSpaceTypes_thenReturnsScore() {
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv = new ArrayList<>();
        qv.add(1.0);
        qv.add(0.0);
        queryVectors.add(qv);

        List<List<Number>> docVectors = new ArrayList<>();
        List<Number> dv = new ArrayList<>();
        dv.add(1.0);
        dv.add(0.0);
        docVectors.add(dv);

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        String[] supportedSpaceTypes = { "innerproduct", "cosinesimil", "l2" };

        for (String spaceType : supportedSpaceTypes) {
            float score = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, spaceType);
            assertTrue("Score should be finite for " + spaceType, Float.isFinite(score));
        }
    }

    /**
     * Tests late interaction score with multiple query and document vectors.
     */
    public void testLateInteractionScore_whenMultipleVectors_thenReturnsSum() {
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv1 = new ArrayList<>();
        qv1.add(1.0);
        qv1.add(0.0);
        queryVectors.add(qv1);

        List<Number> qv2 = new ArrayList<>();
        qv2.add(0.0);
        qv2.add(1.0);
        queryVectors.add(qv2);

        List<List<Number>> docVectors = new ArrayList<>();
        List<Number> dv1 = new ArrayList<>();
        dv1.add(1.0);
        dv1.add(0.0);
        docVectors.add(dv1);

        List<Number> dv2 = new ArrayList<>();
        dv2.add(0.0);
        dv2.add(1.0);
        docVectors.add(dv2);

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        float score = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "innerproduct");
        assertTrue("Score should be positive for multiple vectors", score > 0);
    }

    /**
     * Tests late interaction score with default space type.
     */
    public void testLateInteractionScore_whenDefaultSpaceType_thenUsesL2() {
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv = new ArrayList<>();
        qv.add(1.0);
        qv.add(0.0);
        queryVectors.add(qv);

        List<List<Number>> docVectors = new ArrayList<>();
        List<Number> dv = new ArrayList<>();
        dv.add(1.0);
        dv.add(0.0);
        docVectors.add(dv);

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        float defaultScore = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc);
        float l2Score = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "l2");

        assertEquals("Default should use L2", defaultScore, l2Score, 0.001f);
    }
}
