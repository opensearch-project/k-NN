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
        // Create query vectors
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        queryVectors.add(new float[] { 0.5f, 0.6f, 0.7f, 0.8f });

        // Create document vectors
        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        docVectors.add(new float[] { 0.5f, 0.6f, 0.7f, 0.8f });

        // Create document source
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Calculate expected result
        // Let's calculate all dot products:
        // double qv1dv1 = 0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4; // = 0.3
        // double qv1dv2 = 0.1*0.5 + 0.2*0.6 + 0.3*0.7 + 0.4*0.8; // = 0.7
        // double qv2dv1 = 0.5*0.1 + 0.6*0.2 + 0.7*0.3 + 0.8*0.4; // = 0.7
        // double qv2dv2 = 0.5*0.5 + 0.6*0.6 + 0.7*0.7 + 0.8*0.8; // = 1.74

        // For qv1, max similarity is with dv2: 0.7
        // For qv2, max similarity is with dv2: 1.74
        // Total: 0.7 + 1.74 = 2.44
        double expected = 2.44;

        // Calculate actual result
        double actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc);

        // Assert
        assertEquals(expected, actual, 0.001);
    }

    /**
     * Tests late interaction score with empty or null input vectors.
     * Verifies that the function returns zero for edge cases.
     */
    public void testLateInteractionScore_whenEmptyVectors_thenReturnsZero() {
        // Test with empty query vectors
        List<float[]> emptyQueryVectors = new ArrayList<>();
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", new ArrayList<float[]>());

        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(emptyQueryVectors, "my_vector", doc)
        );
        assertTrue(exception.getMessage().contains("Query vectors cannot be empty"));

        // Test with missing field
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 0.1f });

        assertEquals(0.0, KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "non_existent_field", doc), 0.0);
    }

    /**
     * Tests late interaction score with null or empty individual query vectors.
     * Verifies that appropriate exception is thrown for null/empty vectors within the list.
     */
    public void testLateInteractionScore_whenNullOrEmptyIndividualVectors_thenThrowsException() {
        Map<String, Object> doc = new HashMap<>();
        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 1.0f, 2.0f });
        doc.put("my_vector", docVectors);

        // Test with null vector in query list
        List<float[]> queryVectorsWithNull = new ArrayList<>();
        queryVectorsWithNull.add(null);

        IllegalArgumentException exception1 = expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectorsWithNull, "my_vector", doc)
        );
        assertTrue(exception1.getMessage().contains("Every single vector within query vectors cannot be empty or null"));

        // Test with empty vector in query list
        List<float[]> queryVectorsWithEmpty = new ArrayList<>();
        queryVectorsWithEmpty.add(new float[] {});

        IllegalArgumentException exception2 = expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectorsWithEmpty, "my_vector", doc)
        );
        assertTrue(exception2.getMessage().contains("Every single vector within query vectors cannot be empty or null"));
    }

    /**
     * Tests late interaction score with mismatched vector dimensions.
     * Verifies that appropriate exception is thrown for incompatible vectors.
     */
    public void testLateInteractionScore_whenDimensionMismatch_thenThrowsException() {
        // Create query vectors with 2 dimensions
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 0.1f, 0.2f });

        // Create document vectors with 3 dimensions (mismatch)
        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 0.1f, 0.2f, 0.3f });

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
     * Verifies that maximum similarity is correctly identified across multiple candidates.
     */
    public void testLateInteractionScore_whenMultipleDocVectors_thenReturnsMaxSimilaritySum() {
        // Create query vectors
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 0.1f, 0.2f });

        // Create multiple document vectors
        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 0.1f, 0.2f });
        docVectors.add(new float[] { 0.3f, 0.4f });
        docVectors.add(new float[] { 0.5f, 0.6f });

        // Create document source
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Calculate expected result
        // For qv, max similarity with doc vectors is with dv3: 0.1*0.5 + 0.2*0.6 = 0.17
        double expected = 0.17;

        // Calculate actual result
        double actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc);

        // Assert
        assertEquals(expected, actual, 0.001);
    }

    /**
     * Tests late interaction score with inner product space type.
     * Verifies correct calculation using dot product similarity.
     */
    public void testLateInteractionScore_whenInnerProduct_thenReturnsCorrectScore() {
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 1.0f, 2.0f });

        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 3.0f, 4.0f });

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Expected: 1.0*3.0 + 2.0*4.0 = 11.0
        double expected = 11.0;
        double actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "innerproduct");

        assertEquals(expected, actual, 0.001);
    }

    /**
     * Tests late interaction score with cosine similarity space type.
     * Verifies correct calculation using cosine similarity.
     */
    public void testLateInteractionScore_whenCosinesimil_thenReturnsCorrectScore() {
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 1.0f, 0.0f });

        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 1.0f, 0.0f });

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Expected: cosine similarity of identical unit vectors = 1.0
        double expected = 1.0;
        double actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "cosinesimil");

        assertEquals(expected, actual, 0.001);
    }

    /**
     * Tests late interaction score with L2 distance space type.
     * Verifies correct calculation and score transformation for distance metric.
     */
    public void testLateInteractionScore_whenL2_thenReturnsCorrectScore() {
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 1.0f, 1.0f });

        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 1.0f, 1.0f });

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Expected: L2 distance = 0, transformed score = 1/(1+0) = 1.0
        double expected = 1.0;
        double actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "l2");

        assertEquals(expected, actual, 0.001);
    }

    /**
     * Tests late interaction score with L1 distance space type.
     * Verifies correct calculation and score transformation for distance metric.
     */
    public void testLateInteractionScore_whenL1_thenReturnsCorrectScore() {
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 2.0f, 3.0f });

        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 1.0f, 1.0f });

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Expected: L1 distance = |2-1| + |3-1| = 3, transformed score = 1/(1+3) = 0.25
        double expected = 0.25;
        double actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "l1");

        assertEquals(expected, actual, 0.001);
    }

    /**
     * Tests late interaction score with L-infinity distance space type.
     * Verifies correct calculation and score transformation for distance metric.
     */
    public void testLateInteractionScore_whenLinf_thenReturnsCorrectScore() {
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 5.0f, 2.0f });

        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 3.0f, 1.0f });

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        // Expected: L-inf distance = max(|5-3|, |2-1|) = max(2, 1) = 2, transformed score = 1/(1+2) = 0.333
        double expected = 0.333;
        double actual = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "linf");

        assertEquals(expected, actual, 0.001);
    }

    /**
     * Tests validation of empty query vectors.
     * Verifies that appropriate exception is thrown for empty query vector list.
     */
    public void testLateInteractionScore_whenEmptyQueryVectors_thenThrowsException() {
        List<float[]> emptyQueryVectors = new ArrayList<>();
        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", new ArrayList<float[]>());

        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(emptyQueryVectors, "my_vector", doc)
        );

        assertTrue(exception.getMessage().contains("Query vectors cannot be empty"));
    }

    /**
     * Tests validation of inconsistent query vector dimensions.
     * Verifies that appropriate exception is thrown when query vectors have different dimensions.
     */
    public void testLateInteractionScore_whenInconsistentQueryVectorDimensions_thenThrowsException() {
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 1.0f, 2.0f });        // 2 dimensions
        queryVectors.add(new float[] { 1.0f, 2.0f, 3.0f });  // 3 dimensions - mismatch

        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 1.0f, 2.0f });

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc)
        );

        assertTrue(exception.getMessage().contains("Query vector dimension mismatch"));
        assertTrue(exception.getMessage().contains("expected 2 dimensions, but found 3 dimensions"));
    }

    /**
     * Tests validation of inconsistent document vector dimensions.
     * Verifies that appropriate exception is thrown when document vectors have different dimensions.
     */
    public void testLateInteractionScore_whenInconsistentDocVectorDimensions_thenThrowsException() {
        List<float[]> queryVectors = new ArrayList<>();
        queryVectors.add(new float[] { 1.0f, 2.0f });

        List<float[]> docVectors = new ArrayList<>();
        docVectors.add(new float[] { 1.0f, 2.0f });           // 2 dimensions
        docVectors.add(new float[] { 1.0f, 2.0f, 3.0f });     // 3 dimensions - mismatch

        Map<String, Object> doc = new HashMap<>();
        doc.put("my_vector", docVectors);

        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc)
        );

        assertTrue(exception.getMessage().contains("Document vector dimension mismatch"));
        assertTrue(exception.getMessage().contains("expected 2 dimensions, but found 3 dimensions"));
    }
}
