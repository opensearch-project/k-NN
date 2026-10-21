/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.index.VectorSimilarityFunction;
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

    /**
     * Two query tokens q1 = (1, 0, 0) and q2 = (0, 1, 0), unit document tokens. Doc A matches q1 exactly and is
     * orthogonal to q2 (best cosines 1.0 and 0.0). Doc B has a single token at cosine 0.55 to both (best cosines
     * 0.55 and 0.55). MaxSim prefers B: sum of best cosines is 1.1 against 1.0, and the sum of minimum squared L2
     * distances is 1.8 against 2.0. Summing Lucene's per-token 1 / (1 + d^2) instead ranks A first (1.333 > 1.053).
     */
    public void testLateInteractionScore_whenL2_thenRanksBySumOfMinSquaredDistances() {
        List<List<Number>> queryVectors = List.of(vector(1, 0, 0), vector(0, 1, 0));
        Map<String, Object> docA = Map.of("my_vector", List.of(vector(1, 0, 0)));
        Map<String, Object> docB = Map.of("my_vector", List.of(vector(0.55, 0.55, Math.sqrt(1 - 2 * 0.55 * 0.55))));

        float scoreA = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", docA, "l2");
        float scoreB = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", docB, "l2");

        // n_q / (1 + sum(min d^2) / n_q): A = 2 / (1 + 2.0 / 2), B = 2 / (1 + 1.8 / 2)
        assertEquals(1.0f, scoreA, 1e-5f);
        assertEquals(2.0f / 1.9f, scoreB, 1e-5f);
        assertTrue("Doc B has the smaller sum of min squared distances and must rank first", scoreB > scoreA);
    }

    /**
     * Query tokens q1 = (1, 0) and q2 = (0, 1). Doc A's best dot products are (0.9, -0.6), MaxSim 0.3. Doc B's are
     * (0.2, 0.2), MaxSim 0.4, so B must rank first. Summing Lucene's per-token scaled inner product instead gives
     * A = 1.9 + 1 / 1.6 = 2.525 and B = 2.4, ranking A first.
     */
    public void testLateInteractionScore_whenInnerProductWithNegativeTokenMax_thenRanksByRawMaxSim() {
        List<List<Number>> queryVectors = List.of(vector(1, 0), vector(0, 1));
        Map<String, Object> docA = Map.of("my_vector", List.of(vector(0.9, -0.6)));
        Map<String, Object> docB = Map.of("my_vector", List.of(vector(0.2, 0.2)));

        float scoreA = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", docA, "innerproduct");
        float scoreB = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", docB, "innerproduct");

        // n_q * scaleMaxInnerProductScore(MaxSim / n_q): A = 2 * (0.15 + 1), B = 2 * (0.2 + 1)
        assertEquals(2.3f, scoreA, 1e-5f);
        assertEquals(2.4f, scoreB, 1e-5f);
        assertTrue("Doc B has the larger raw MaxSim and must rank first", scoreB > scoreA);
    }

    /**
     * A negative total MaxSim must still produce a positive score, because script_score rejects negative scores.
     */
    public void testLateInteractionScore_whenInnerProductMaxSimNegative_thenScoreIsPositive() {
        List<List<Number>> queryVectors = List.of(vector(1, 0), vector(0, 1));
        Map<String, Object> doc = Map.of("my_vector", List.of(vector(-0.5, -0.3)));

        float score = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", doc, "innerproduct");

        // MaxSim = -0.8, so 2 * 1 / (1 + 0.4)
        assertEquals(2.0f / 1.4f, score, 1e-5f);
    }

    /**
     * Cosine is the control: the per-token transform (1 + cos) / 2 is affine, so the score is (n_q + MaxSim) / 2
     * both before and after the fix.
     */
    public void testLateInteractionScore_whenCosine_thenScoreIsAffineInMaxSim() {
        List<List<Number>> queryVectors = List.of(vector(1, 0, 0), vector(0, 1, 0));
        Map<String, Object> docA = Map.of("my_vector", List.of(vector(1, 0, 0)));
        Map<String, Object> docB = Map.of("my_vector", List.of(vector(0.55, 0.55, Math.sqrt(1 - 2 * 0.55 * 0.55))));

        float scoreA = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", docA, "cosinesimil");
        float scoreB = KNNPainlessScriptUtils.lateInteractionScore(queryVectors, "my_vector", docB, "cosinesimil");

        assertEquals((2 + 1.0f) / 2, scoreA, 1e-5f);
        assertEquals((2 + 1.1f) / 2, scoreB, 1e-5f);
        assertTrue(scoreB > scoreA);
    }

    /**
     * With a single query token the score is unchanged from the per-token Lucene score, and with every per-token
     * inner product maximum non-negative the score equals the old sum of per-token scaled scores.
     */
    public void testLateInteractionScore_whenSingleQueryTokenOrNonNegativeInnerProduct_thenMatchesLuceneScore() {
        List<Number> q = vector(0.3, -0.4);
        List<Number> d = vector(0.1, 0.2);
        float[] qArr = { 0.3f, -0.4f };
        float[] dArr = { 0.1f, 0.2f };
        Map<String, Object> doc = Map.of("my_vector", List.of(d));

        assertEquals(
            VectorSimilarityFunction.EUCLIDEAN.compare(qArr, dArr),
            KNNPainlessScriptUtils.lateInteractionScore(List.of(q), "my_vector", doc, "l2"),
            1e-6f
        );
        assertEquals(
            VectorSimilarityFunction.COSINE.compare(qArr, dArr),
            KNNPainlessScriptUtils.lateInteractionScore(List.of(q), "my_vector", doc, "cosinesimil"),
            1e-6f
        );
        assertEquals(
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(qArr, dArr),
            KNNPainlessScriptUtils.lateInteractionScore(List.of(q), "my_vector", doc, "innerproduct"),
            1e-6f
        );

        // Best dot products 0.9 and 0.2, both non-negative: (0.9 + 1) + (0.2 + 1)
        Map<String, Object> twoTokenDoc = Map.of("my_vector", List.of(vector(0.9, 0.0), vector(0.0, 0.2)));
        assertEquals(
            3.1f,
            KNNPainlessScriptUtils.lateInteractionScore(List.of(vector(1, 0), vector(0, 1)), "my_vector", twoTokenDoc, "innerproduct"),
            1e-5f
        );
    }

    private static List<Number> vector(double... values) {
        List<Number> vector = new ArrayList<>(values.length);
        for (double value : values) {
            vector.add(value);
        }
        return vector;
    }
}
