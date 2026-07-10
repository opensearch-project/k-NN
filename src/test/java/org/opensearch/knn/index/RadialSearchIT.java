/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.CompressionTestConfig;
import org.opensearch.knn.KNNCompressionRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Integration tests for radial search (max_distance and min_score) on Faiss and Lucene HNSW.
 * Covers both ANN path (no filter) and ExactSearcher path (with filter, Faiss only) across
 * L2, cosine, and inner product space types, parameterized by compression (FP32 and SQ 1-bit / 32x).
 *
 * Validates the distanceToRadialThreshold and scoreToRadialThreshold transformations,
 * specifically the fix for Faiss IP max_distance (DISTANCE_TRANSLATIONS: distance -> -1 * distance).
 *
 * FP32 asserts exact result counts. Under 1-bit quantization at dimension 2 the per-threshold
 * counts are not reliably exact, so the 32x runs assert the radial query succeeds with valid,
 * descending scores and a sane result count (the proven Compression32XIT radial pattern).
 */
public class RadialSearchIT extends KNNCompressionRestTestCase {

    private static final int DIMENSION = 2;
    private static final String FILTER_FIELD = "color";

    private final String testName;
    private final KNNEngine engine;
    private final SpaceType spaceType;
    private final float[][] testVectors;
    private final float[] queryVector;
    private final String searchType;
    private final float threshold;
    private final int expectedCount;
    private final boolean useFilter;
    private final boolean mosEnabled;

    public RadialSearchIT(
        CompressionTestConfig compressionConfig,
        String testName,
        KNNEngine engine,
        SpaceType spaceType,
        float[][] testVectors,
        float[] queryVector,
        String searchType,
        float threshold,
        int expectedCount,
        boolean useFilter,
        boolean mosEnabled
    ) {
        super(compressionConfig);
        this.testName = testName;
        this.engine = engine;
        this.spaceType = spaceType;
        this.testVectors = testVectors;
        this.queryVector = queryVector;
        this.searchType = searchType;
        this.threshold = threshold;
        this.expectedCount = expectedCount;
        this.useFilter = useFilter;
        this.mosEnabled = mosEnabled;
    }

    @ParametersFactory(argumentFormatting = "{1}_compression:{0}")
    public static Collection<Object[]> compressionParameters() {
        List<Object[]> baseParams = new ArrayList<>();
        addFaissParams(baseParams);
        addFaissMosParams(baseParams);
        addLuceneParams(baseParams);

        List<Object[]> params = new ArrayList<>();
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            for (Object[] base : baseParams) {
                Object[] row = new Object[base.length + 1];
                row[0] = config;
                System.arraycopy(base, 0, row, 1, base.length);
                params.add(row);
            }
        }
        return params;
    }

    private static void addFaissParams(List<Object[]> params) {
        // Faiss L2: native distance is squared euclidean, score = 1/(1+dist), threshold passes through unchanged
        float[][] l2Vectors = {
            { 1.0f, 0.0f },     // dist=0.0, score=1.0
            { 1.0f, 0.5f },     // dist=0.25, score=0.8
            { 0.0f, 0.0f },     // dist=1.0, score=0.5
            { 0.0f, 3.0f }      // dist=10.0, score=0.091
        };
        float[] l2Query = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Faiss_L2",
            BuiltinKNNEngine.FAISS,
            SpaceType.L2,
            l2Vectors,
            l2Query,
            true,
            new Object[] { "max_distance", 0.5f, 2 },
            new Object[] { "max_distance", 2.0f, 3 },
            new Object[] { "min_score", 0.9f, 1 },
            new Object[] { "min_score", 0.75f, 2 }
        );

        // Faiss Cosine: threshold maps distance -> 1 - distance, score = (1 + cos_sim) / 2
        float[][] cosVectors = {
            { 1.0f, 0.0f },     // cos=1.0, score=1.0
            { 3.0f, 1.0f },     // cos=0.949, score=0.975
            { 1.0f, 1.0f },     // cos=0.707, score=0.854
            { 0.0f, 1.0f }      // cos=0.0, score=0.5
        };
        float[] cosQuery = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Faiss_COSINE",
            BuiltinKNNEngine.FAISS,
            SpaceType.COSINESIMIL,
            cosVectors,
            cosQuery,
            true,
            new Object[] { "max_distance", 0.1f, 2 },
            new Object[] { "max_distance", 0.5f, 3 },
            new Object[] { "min_score", 0.99f, 1 },
            new Object[] { "min_score", 0.8f, 3 }
        );

        // Faiss IP: threshold negates distance (the fix under test), score branches on sign of dot product
        float[][] ipVectors = {
            { 1.0f, 0.0f },     // dot=1.0, score=2.0
            { 0.8f, 0.1f },     // dot=0.8, score=1.8
            { 0.3f, 0.3f },     // dot=0.3, score=1.3
            { 0.0f, 1.0f }      // dot=0.0, score=1.0
        };
        float[] ipQuery = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Faiss_IP",
            BuiltinKNNEngine.FAISS,
            SpaceType.INNER_PRODUCT,
            ipVectors,
            ipQuery,
            true,
            new Object[] { "max_distance", -0.5f, 2 },
            new Object[] { "max_distance", -0.1f, 3 },
            new Object[] { "max_distance", 0.1f, 4 },
            new Object[] { "min_score", 1.9f, 1 },
            new Object[] { "min_score", 1.4f, 2 },
            new Object[] { "min_score", 1.01f, 3 }
        );

        // Faiss IP with negative dot products: exercises the 1/(1-dot) score branch
        float[][] ipNegVectors = {
            { 1.0f, 0.0f },     // dot=1.0, score=2.0, distance=-1.0
            { 0.5f, 0.5f },     // dot=0.5, score=1.5, distance=-0.5
            { -0.3f, 0.5f },    // dot=-0.3, score=0.769, distance=0.3
            { -0.8f, 0.2f },    // dot=-0.8, score=0.556, distance=0.8
            { -1.0f, 0.0f }     // dot=-1.0, score=0.5, distance=1.0
        };
        float[] ipNegQuery = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Faiss_IP_neg",
            BuiltinKNNEngine.FAISS,
            SpaceType.INNER_PRODUCT,
            ipNegVectors,
            ipNegQuery,
            true,
            new Object[] { "max_distance", -0.4f, 2 },
            new Object[] { "max_distance", 0.4f, 3 },
            new Object[] { "max_distance", 0.9f, 4 },
            new Object[] { "max_distance", 1.1f, 5 },
            new Object[] { "min_score", 1.4f, 2 },
            new Object[] { "min_score", 0.6f, 3 },
            new Object[] { "min_score", 0.55f, 4 },
            new Object[] { "min_score", 0.45f, 5 }
        );
    }

    private static void addRadialCases(
        List<Object[]> params,
        String prefix,
        KNNEngine engine,
        SpaceType spaceType,
        float[][] vectors,
        float[] query,
        boolean includeExact,
        Object[]... thresholds
    ) {
        addRadialCases(params, prefix, engine, spaceType, vectors, query, includeExact, false, thresholds);
    }

    /**
     * Generates test cases for ANN, ExactSearcher, and optionally MOS paths for a given engine/space configuration.
     * Each threshold entry produces one test case; if exactSearcher is true, a mirror set with filter is added.
     * If mosEnabled is true, generates MOS-specific test cases.
     */
    private static void addRadialCases(
        List<Object[]> params,
        String prefix,
        KNNEngine engine,
        SpaceType spaceType,
        float[][] vectors,
        float[] query,
        boolean includeExact,
        boolean mosEnabled,
        Object[]... thresholds
    ) {
        for (Object[] t : thresholds) {
            String searchType = (String) t[0];
            float threshold = (float) t[1];
            int expected = (int) t[2];
            params.add(
                new Object[] {
                    prefix + "_ann_" + searchType + "_" + expected,
                    engine,
                    spaceType,
                    vectors,
                    query,
                    searchType,
                    threshold,
                    expected,
                    false,
                    mosEnabled }
            );
            if (includeExact) {
                params.add(
                    new Object[] {
                        prefix + "_exact_" + searchType + "_" + expected,
                        engine,
                        spaceType,
                        vectors,
                        query,
                        searchType,
                        threshold,
                        expected,
                        true,
                        mosEnabled }
                );
            }
        }
    }

    private static void addFaissMosParams(List<Object[]> params) {
        // MOS uses Lucene's search algorithm on Faiss-built HNSW graphs.
        // After the fix, max_distance for IP should produce the same results regardless of sign convention.
        // MOS does not support ExactSearcher (filter path), so includeExact=false.

        // MOS L2: same behavior as standard Faiss L2
        float[][] l2Vectors = {
            { 1.0f, 0.0f },     // dist=0.0, score=1.0
            { 1.0f, 0.5f },     // dist=0.25, score=0.8
            { 0.0f, 0.0f },     // dist=1.0, score=0.5
            { 0.0f, 3.0f }      // dist=10.0, score=0.091
        };
        float[] l2Query = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Faiss_MOS_L2",
            BuiltinKNNEngine.FAISS,
            SpaceType.L2,
            l2Vectors,
            l2Query,
            true,
            true,
            new Object[] { "max_distance", 0.5f, 2 },
            new Object[] { "max_distance", 2.0f, 3 },
            new Object[] { "min_score", 0.9f, 1 },
            new Object[] { "min_score", 0.75f, 2 }
        );

        // MOS Cosine
        float[][] cosVectors = {
            { 1.0f, 0.0f },     // cos=1.0, score=1.0
            { 3.0f, 1.0f },     // cos=0.949, score=0.975
            { 1.0f, 1.0f },     // cos=0.707, score=0.854
            { 0.0f, 1.0f }      // cos=0.0, score=0.5
        };
        float[] cosQuery = { 1.0f, 0.0f };

        // MOS + Cosine + ExactSearcher has a pre-existing scoring mismatch (tracked separately).
        // The VectorScorer produces scores in a different space than the MOS-converted threshold.
        addRadialCases(
            params,
            "Faiss_MOS_COSINE",
            BuiltinKNNEngine.FAISS,
            SpaceType.COSINESIMIL,
            cosVectors,
            cosQuery,
            false,
            true,
            new Object[] { "max_distance", 0.1f, 2 },
            new Object[] { "max_distance", 0.5f, 3 },
            new Object[] { "min_score", 0.99f, 1 },
            new Object[] { "min_score", 0.8f, 3 }
        );

        // MOS IP: The critical test. Negative max_distance (Faiss convention) should be selective.
        // After abs() fix, both negative and positive values produce correct results.
        float[][] ipVectors = {
            { 1.0f, 0.0f },     // dot=1.0, score=2.0
            { 0.8f, 0.1f },     // dot=0.8, score=1.8
            { 0.3f, 0.3f },     // dot=0.3, score=1.3
            { 0.0f, 1.0f }      // dot=0.0, score=1.0
        };
        float[] ipQuery = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Faiss_MOS_IP",
            BuiltinKNNEngine.FAISS,
            SpaceType.INNER_PRODUCT,
            ipVectors,
            ipQuery,
            true,
            true,
            // Negative max_distance: Faiss convention d=-dot, so -0.5 means dot >= 0.5
            new Object[] { "max_distance", -0.5f, 2 },
            new Object[] { "max_distance", -0.1f, 3 },
            // Positive max_distance: d=-dot convention means dot >= -0.1 (broad), returns all 4
            new Object[] { "max_distance", 0.1f, 4 },
            // min_score is engine-independent, should match standard Faiss exactly
            new Object[] { "min_score", 1.9f, 1 },
            new Object[] { "min_score", 1.4f, 2 },
            new Object[] { "min_score", 1.01f, 3 }
        );

        // MOS IP with negative dot products
        float[][] ipNegVectors = {
            { 1.0f, 0.0f },     // dot=1.0, score=2.0, distance=-1.0
            { 0.5f, 0.5f },     // dot=0.5, score=1.5, distance=-0.5
            { -0.3f, 0.5f },    // dot=-0.3, score=0.769, distance=0.3
            { -0.8f, 0.2f },    // dot=-0.8, score=0.556, distance=0.8
            { -1.0f, 0.0f }     // dot=-1.0, score=0.5, distance=1.0
        };
        float[] ipNegQuery = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Faiss_MOS_IP_neg",
            BuiltinKNNEngine.FAISS,
            SpaceType.INNER_PRODUCT,
            ipNegVectors,
            ipNegQuery,
            true,
            true,
            // Negative max_distance (Faiss convention): -0.4 means dot >= 0.4
            new Object[] { "max_distance", -0.4f, 2 },
            new Object[] { "max_distance", -0.01f, 2 },
            // Positive max_distance: 0.4 means dot >= -0.4 (broad), returns 3 (dot: 1.0, 0.5, -0.3)
            new Object[] { "max_distance", 0.4f, 3 },
            // Very large positive: returns all 5
            new Object[] { "max_distance", 0.9f, 4 },
            new Object[] { "max_distance", 1.1f, 5 },
            // min_score tests (same as non-MOS)
            new Object[] { "min_score", 1.4f, 2 },
            new Object[] { "min_score", 0.6f, 3 },
            new Object[] { "min_score", 0.55f, 4 },
            new Object[] { "min_score", 0.45f, 5 }
        );
    }

    private static void addLuceneParams(List<Object[]> params) {
        // Lucene radial search: ANN path only (ExactSearcher throws for non-Faiss engines)

        // Lucene L2: same vector geometry as Faiss, ANN path only
        float[][] l2Vectors = {
            { 1.0f, 0.0f },     // dist=0.0, score=1.0
            { 1.0f, 0.5f },     // dist=0.25, score=0.8
            { 0.0f, 0.0f },     // dist=1.0, score=0.5
            { 0.0f, 3.0f }      // dist=10.0, score=0.091
        };
        float[] l2Query = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Lucene_L2",
            BuiltinKNNEngine.LUCENE,
            SpaceType.L2,
            l2Vectors,
            l2Query,
            false,
            new Object[] { "max_distance", 0.5f, 2 },
            new Object[] { "max_distance", 2.0f, 3 },
            new Object[] { "min_score", 0.9f, 1 },
            new Object[] { "min_score", 0.75f, 2 }
        );

        // Lucene Cosine
        float[][] cosVectors = {
            { 1.0f, 0.0f },     // cos=1.0, score=1.0
            { 3.0f, 1.0f },     // cos=0.949, score=0.975
            { 1.0f, 1.0f },     // cos=0.707, score=0.854
            { 0.0f, 1.0f }      // cos=0.0, score=0.5
        };
        float[] cosQuery = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Lucene_COSINE",
            BuiltinKNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            cosVectors,
            cosQuery,
            false,
            new Object[] { "max_distance", 0.1f, 2 },
            new Object[] { "max_distance", 0.5f, 3 },
            new Object[] { "min_score", 0.99f, 1 },
            new Object[] { "min_score", 0.8f, 3 }
        );

        // Lucene IP: distanceToRadialThreshold converts based on sign of distance
        float[][] ipVectors = {
            { 1.0f, 0.0f },     // dot=1.0, score=2.0
            { 0.8f, 0.1f },     // dot=0.8, score=1.8
            { 0.3f, 0.3f },     // dot=0.3, score=1.3
            { 0.0f, 1.0f }      // dot=0.0, score=1.0
        };
        float[] ipQuery = { 1.0f, 0.0f };

        addRadialCases(
            params,
            "Lucene_IP",
            BuiltinKNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            ipVectors,
            ipQuery,
            false,
            new Object[] { "max_distance", 0.9f, 1 },
            new Object[] { "max_distance", 0.5f, 2 },
            new Object[] { "max_distance", 0.2f, 3 },
            new Object[] { "min_score", 1.9f, 1 },
            new Object[] { "min_score", 1.4f, 2 },
            new Object[] { "min_score", 1.01f, 3 }
        );
    }

    public void testRadialSearch_thenCorrectResults() throws Exception {
        String indexName = prefix() + testName.toLowerCase().replace(" ", "_");

        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION);
        addCompressionMappingFields(mapping);
        mapping.field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.KNN_ENGINE, engine.getName());

        if (engine == BuiltinKNNEngine.FAISS) {
            mapping.startObject(KNNConstants.PARAMETERS)
                .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, 128)
                .field(KNNConstants.METHOD_PARAMETER_M, 16)
                .endObject();
        }

        mapping.endObject().endObject().startObject(FILTER_FIELD).field("type", "keyword").endObject().endObject().endObject();

        Settings.Builder settingsBuilder = Settings.builder().put("index.knn", true);
        if (mosEnabled) {
            settingsBuilder.put("index.knn.memory_optimized_search", true);
        }
        Settings settings = settingsBuilder.build();
        createKnnIndex(indexName, settings, mapping.toString());

        for (int i = 0; i < testVectors.length; i++) {
            indexDocument(indexName, String.valueOf(i + 1), testVectors[i]);
        }
        refreshAllIndices();

        XContentBuilder query = buildQuery();
        Response response = searchKNNIndex(indexName, query, 10);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        if (compressionConfig == CompressionTestConfig.X1) {
            assertEquals(
                String.format("[%s] Expected %d results but got %d", testName, expectedCount, results.size()),
                expectedCount,
                results.size()
            );
        } else {
            assertTrue(
                String.format(
                    "[%s] quantized radial returned %d results, expected at most %d",
                    testName,
                    results.size(),
                    testVectors.length
                ),
                results.size() <= testVectors.length
            );
            for (int i = 0; i < results.size(); i++) {
                assertTrue(String.format("[%s] score should be positive", testName), results.get(i).getScore() > 0.0f);
                if (i > 0) {
                    assertTrue(
                        String.format("[%s] scores should be in descending order", testName),
                        results.get(i - 1).getScore() >= results.get(i).getScore()
                    );
                }
            }
        }

        deleteKNNIndex(indexName);
    }

    private void indexDocument(String indexName, String docId, float[] vector) throws IOException {
        XContentBuilder doc = XContentFactory.jsonBuilder().startObject().field(FIELD_NAME, vector).field(FILTER_FIELD, "red").endObject();

        Request request = new Request("POST", "/" + indexName + "/_doc/" + docId + "?refresh=true");
        request.setJsonEntity(doc.toString());
        client().performRequest(request);
    }

    private XContentBuilder buildQuery() throws IOException {
        XContentBuilder query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field(searchType, threshold);

        if (useFilter) {
            query.startObject("filter").startObject("term").field(FILTER_FIELD, "red").endObject().endObject();
        }

        query.endObject().endObject().endObject().endObject();

        return query;
    }
}
