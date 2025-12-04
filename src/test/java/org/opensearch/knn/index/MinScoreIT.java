/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import com.google.common.primitives.Floats;
import org.apache.http.util.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Integration tests for min_score filtering with different KNN engines.
 * Tests the score-to-distance transformations for various space types.
 */
public class MinScoreIT extends KNNRestTestCase {

    private final KNNEngine knnEngine;
    private final SpaceType spaceType;
    private final float[][] testVectors;
    private final float[] queryVector;
    private final float[] minScores;
    private final int[] expectedCounts;

    public MinScoreIT(
        KNNEngine knnEngine,
        SpaceType spaceType,
        float[][] testVectors,
        float[] queryVector,
        float[] minScores,
        int[] expectedCounts
    ) {
        this.knnEngine = knnEngine;
        this.spaceType = spaceType;
        this.testVectors = testVectors;
        this.queryVector = queryVector;
        this.minScores = minScores;
        this.expectedCounts = expectedCounts;
    }

    /**
     * Test scenario encapsulating test data for a specific space type.
     */
    private static class TestScenario {
        final SpaceType spaceType;
        final float[][] vectors;
        final float[] query;
        final float[] minScores;
        final int[] expectedCounts;

        TestScenario(SpaceType spaceType, float[][] vectors, float[] query, float[] minScores, int[] expectedCounts) {
            this.spaceType = spaceType;
            this.vectors = vectors;
            this.query = query;
            this.minScores = minScores;
            this.expectedCounts = expectedCounts;
        }
    }

    /**
     * Inner Product test scenario with standard positive values.
     */
    private static TestScenario innerProductScenario() {
        float[][] vectors = {
            { 1.0f, 0.0f },     // IP=1.0, score=2.0
            { 0.8f, 0.1f },     // IP=0.8, score=1.8
            { 0.5f, 0.5f },     // IP=0.5, score=1.5
            { 0.2f, 0.8f },     // IP=0.2, score=1.2
            { 0.0f, 1.0f }      // IP=0.0, score=1.0 (orthogonal)
        };
        float[] query = { 1.0f, 0.0f };
        float[] minScores = { 1.4f, 1.01f, 1.9f, 0.5f };
        int[] expectedCounts = { 3, 4, 1, 5 };
        return new TestScenario(SpaceType.INNER_PRODUCT, vectors, query, minScores, expectedCounts);
    }

    /**
     * Cosine Similarity test scenario.
     */
    private static TestScenario cosineScenario() {
        float[][] vectors = {
            { 1.0f, 0.0f },     // cosine=1.0, score=1.0
            { 0.8f, 0.6f },     // cosine=0.8, score=0.9
            { 0.6f, 0.8f },     // cosine=0.6, score=0.8
            { 0.0f, 1.0f },     // cosine=0.0, score=0.5 (orthogonal)
            { -1.0f, 0.0f }     // cosine=-1.0, score=0.0 (opposite)
        };
        float[] query = { 1.0f, 0.0f };
        float[] minScores = { 0.95f, 0.85f, 0.75f, 0.4f };
        int[] expectedCounts = { 1, 2, 3, 4 };
        return new TestScenario(SpaceType.COSINESIMIL, vectors, query, minScores, expectedCounts);
    }

    /**
     * Inner Product test scenario with negative values.
     */
    private static TestScenario innerProductNegativeScenario() {
        float[][] vectors = {
            { 0.5f, 0.0f },     // IP=0.5, score=1.5
            { 0.0f, 1.0f },     // IP=0.0, score=1.0
            { -0.5f, 0.0f },    // IP=-0.5, score=0.667
            { -1.0f, 0.0f }     // IP=-1.0, score=0.5
        };
        float[] query = { 1.0f, 0.0f };
        float[] minScores = { 0.6f };
        int[] expectedCounts = { 3 };
        return new TestScenario(SpaceType.INNER_PRODUCT, vectors, query, minScores, expectedCounts);
    }

    /**
     * L2 (Euclidean) distance test scenario.
     * Note: Faiss and Lucene use SQUARED L2 distance: distance² = (x1-q1)² + (x2-q2)²
     * Score formula: 1 / (1 + distance²)
     */
    private static TestScenario l2Scenario() {
        float[][] vectors = {
            { 1.0f, 0.0f },     // distance²=0.0, score=1.0 (identical)
            { 1.0f, 0.5f },     // distance²=0.25, score=0.8
            { 1.0f, 1.0f },     // distance²=1.0, score=0.5
            { 0.0f, 0.0f },     // distance²=1.0, score=0.5
            { 0.0f, 2.0f }      // distance²=5.0, score=0.167
        };
        float[] query = { 1.0f, 0.0f };
        float[] minScores = { 0.9f, 0.75f, 0.45f, 0.15f };
        int[] expectedCounts = { 1, 2, 4, 5 };
        return new TestScenario(SpaceType.L2, vectors, query, minScores, expectedCounts);
    }

    /**
     * Generate test parameters for Faiss engine.
     */
    private static List<Object[]> generateFaissTests(List<TestScenario> scenarios) {
        List<Object[]> params = new ArrayList<>();
        for (TestScenario scenario : scenarios) {
            params.add(
                new Object[] {
                    KNNEngine.FAISS,
                    scenario.spaceType,
                    scenario.vectors,
                    scenario.query,
                    scenario.minScores,
                    scenario.expectedCounts }
            );
        }
        return params;
    }

    /**
     * Generate test parameters for Lucene engine.
     */
    private static List<Object[]> generateLuceneTests(List<TestScenario> scenarios) {
        List<Object[]> params = new ArrayList<>();
        for (TestScenario scenario : scenarios) {
            params.add(
                new Object[] {
                    KNNEngine.LUCENE,
                    scenario.spaceType,
                    scenario.vectors,
                    scenario.query,
                    scenario.minScores,
                    scenario.expectedCounts }
            );
        }
        return params;
    }

    @ParametersFactory
    public static Collection<Object[]> parameters() {
        List<TestScenario> scenarios = Arrays.asList(
            innerProductScenario(),
            cosineScenario(),
            innerProductNegativeScenario(),
            l2Scenario()
        );

        List<Object[]> params = new ArrayList<>();
        params.addAll(generateFaissTests(scenarios));
        params.addAll(generateLuceneTests(scenarios));
        return params;
    }

    private static final int DIMENSION = 2;
    private static final String FIELD_NAME = "test_vector";

    public void testMinScore_withDifferentMetrics_thenCorrectlyFiltersResults() throws Exception {
        String indexName = "test_"
            + knnEngine.getName()
            + "_"
            + spaceType.getValue()
            + "_minscore";

        // Create index with specified space type and engine
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION);

        if (knnEngine == KNNEngine.LUCENE) {
            // Lucene engine configuration
            builder.field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
                .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
                .endObject();
        } else {
            // Faiss engine configuration
            builder.field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
                .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
                .startObject(KNNConstants.PARAMETERS)
                .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, 128)
                .field(KNNConstants.METHOD_PARAMETER_M, 16)
                .endObject()
                .endObject();
        }

        builder.endObject().endObject().endObject();

        Settings settings = Settings.builder().put("index.knn", true).build();
        createKnnIndex(indexName, settings, builder.toString());

        // Index test vectors
        for (int i = 0; i < testVectors.length; i++) {
            addKnnDoc(indexName, String.valueOf(i + 1), FIELD_NAME, Floats.asList(testVectors[i]).toArray());
        }

        refreshAllIndices();

        // Test with different min_score thresholds
        for (int i = 0; i < minScores.length; i++) {
            float minScore = minScores[i];
            int expectedCount = expectedCounts[i];

            XContentBuilder query = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", queryVector)
                .field("min_score", minScore)
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            Response response = searchKNNIndex(indexName, query, 10);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

            assertEquals(
                String.format(
                    "[%s, %s] Expected %d results with min_score=%.2f",
                    knnEngine.getName(),
                    spaceType.getValue(),
                    expectedCount,
                    minScore
                ),
                expectedCount,
                results.size()
            );

            // Verify all returned scores are >= min_score
            for (KNNResult result : results) {
                assertTrue(
                    String.format(
                        "[%s, %s] Score %.3f should be >= %.2f",
                        knnEngine.getName(),
                        spaceType.getValue(),
                        result.getScore(),
                        minScore
                    ),
                    result.getScore() >= minScore - 0.001f  // Small epsilon for floating point comparison
                );
            }
        }

        deleteKNNIndex(indexName);
    }
}
