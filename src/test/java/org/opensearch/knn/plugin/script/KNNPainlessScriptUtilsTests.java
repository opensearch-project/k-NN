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
     * Tests late interaction score calculation with valid input vectors.
     * Verifies correct computation of maximum similarity scores across query and document vectors.
     */
    public void testLateInteractionScore_whenValidVectors_thenReturnsCorrectScore() {
        // Create query vectors as List<List<Number>> (can be Double from Painless)
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv1 = new ArrayList<>();
        qv1.add(0.1);
        qv1.add(0.2);
        qv1.add(0.3);
        qv1.add(0.4);
        queryVectors.add(qv1);

        List<Number> qv2 = new ArrayList<>();
        qv2.add(0.5);
        qv2.add(0.6);
        qv2.add(0.7);
        qv2.add(0.8);
        queryVectors.add(qv2);

        // Create document vectors as List<List<Number>>
        List<List<Number>> docVectors = new ArrayList<>();
        List<Number> dv1 = new ArrayList<>();
        dv1.add(0.1);
        dv1.add(0.2);
        dv1.add(0.3);
        dv1.add(0.4);
        docVectors.add(dv1);

        List<Number> dv2 = new ArrayList<>();
        dv2.add(0.5);
        dv2.add(0.6);
        dv2.add(0.7);
        dv2.add(0.8);
        docVectors.add(dv2);

        // Create document source
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Calculate expected result
        // For qv1, max similarity is with dv2: 0.7
        // For qv2, max similarity is with dv2: 1.74
        // Total: 0.7 + 1.74 = 2.44
        float expected = 2.44f;

        // Calculate actual result
        float actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc);

        // Assert
        assertEquals(expected, actual, 0.001f);
    }

    /**
     * Tests late interaction score with empty vectors.
     * Verifies that empty input returns zero score.
     */
    public void testLateInteractionScore_whenEmptyVectors_thenReturnsZero() {
        // Test with empty query vectors
        List<List<Number>> emptyQueryVectors = new ArrayList<>();
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", new ArrayList<List<Number>>());

        assertEquals(0.0f, KNNPainlessScriptUtils.lateInteractionScore(emptyQueryVectors, "my_vector", doc), 0.0f);

        // Test with null document vectors
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv = new ArrayList<>();
        qv.add(0.1);
        qv.add(0.2);
        queryVectors.add(qv);

        Map<String, Object> docWithNull = new HashMap<>();
        docWithNull.put("my_vector", null);

        assertEquals(0.0f, KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", docWithNull), 0.0f);
    }

    /**
     * Tests late interaction score with dimension mismatch.
     * Verifies that dimension mismatch throws appropriate exception.
     */
    public void testLateInteractionScore_whenDimensionMismatch_thenThrowsException() {
        // Create query vectors with 2 dimensions
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv = new ArrayList<>();
        qv.add(0.1);
        qv.add(0.2);
        queryVectors.add(qv);

        // Create document vectors with 3 dimensions (mismatch)
        List<List<Number>> docVectors = new ArrayList<>();
        List<Number> dv = new ArrayList<>();
        dv.add(0.1);
        dv.add(0.2);
        dv.add(0.3);
        docVectors.add(dv);

        // Create document source
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Should throw IllegalArgumentException for dimension mismatch
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc)
        );

        assertTrue(exception.getMessage().contains("Vector dimension mismatch"));
        assertTrue(exception.getMessage().contains("query vector has 2 dimensions"));
        assertTrue(exception.getMessage().contains("document vector has 3 dimensions"));
    }

    /**
     * Tests late interaction score with multiple document vectors.
     * Verifies that the function correctly finds maximum similarity for each query vector.
     */
    public void testLateInteractionScore_whenMultipleDocVectors_thenReturnsMaxSimilaritySum() {
        // Create query vectors
        List<List<Number>> queryVectors = new ArrayList<>();
        List<Number> qv = new ArrayList<>();
        qv.add(0.1);
        qv.add(0.2);
        queryVectors.add(qv);

        // Create multiple document vectors
        List<List<Number>> docVectors = new ArrayList<>();
        List<Number> dv1 = new ArrayList<>();
        dv1.add(0.1);
        dv1.add(0.2);
        docVectors.add(dv1);

        List<Number> dv2 = new ArrayList<>();
        dv2.add(0.3);
        dv2.add(0.4);
        docVectors.add(dv2);

        List<Number> dv3 = new ArrayList<>();
        dv3.add(0.5);
        dv3.add(0.6);
        docVectors.add(dv3);

        // Create document source
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Calculate expected result
        // For qv, max similarity with doc vectors is with dv3: 0.1*0.5 + 0.2*0.6 = 0.17
        float expected = 0.17f;

        // Calculate actual result
        float actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc);

        // Assert
        assertEquals(expected, actual, 0.001f);
    }
}
