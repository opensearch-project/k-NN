/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * REST integration tests for the experimental {@code svs} engine ({@code svs_vamana} method), requiring a
 * node built with {@code -Pknn.sandbox.enabled=true} (sandbox jar bundled + the isolated SVS native lib);
 * excluded from default-build integTest runs. SVS names are written as literals to avoid a test-time
 * dependency on the sandbox module.
 */
public class FaissSVSVamanaIT extends KNNRestTestCase {

    private static final String SVS_ENGINE = "svs";
    private static final String SVS_VAMANA = "svs_vamana";
    private static final int DIMENSION = 3;
    private static final float[][] DOCS = new float[][] { { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.0f, 2.0f }, { 3.0f, 3.0f, 3.0f } };

    @SneakyThrows
    public void testSVSVamana_whenBasicConfiguration_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-basic", SpaceType.L2, builder -> {
            builder.startObject(PARAMETERS).field("degree", 64).endObject();
        });
    }

    @SneakyThrows
    public void testSVSVamana_withSqFp16Encoder_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-sq-fp16", SpaceType.L2, builder -> {
            builder.startObject(PARAMETERS)
                .field("degree", 64)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "sq")
                .startObject(PARAMETERS)
                .field("type", "fp16")
                .endObject()
                .endObject()
                .endObject();
        });
    }

    @SneakyThrows
    public void testSVSVamana_withLvqEncoder_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-lvq", SpaceType.L2, builder -> {
            builder.startObject(PARAMETERS)
                .field("degree", 64)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "lvq")
                .startObject(PARAMETERS)
                .field("primary_bits", 4)
                .field("residual_bits", 4)
                .endObject()
                .endObject()
                .endObject();
        });
    }

    // Below the rough training threshold: the LVQ-fallback rung.
    @SneakyThrows
    public void testSVSVamana_withLeanVecEncoder_smallSegmentFallsBackToLvq_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-leanvec-fallback", SpaceType.L2, builder -> {
            builder.startObject(PARAMETERS)
                .field("degree", 64)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "leanvec")
                .startObject(PARAMETERS)
                .field("primary_bits", 4)
                .field("residual_bits", 8)
                .endObject()
                .endObject()
                .endObject();
        });
    }

    // Flush-time segments stay below the 1000 threshold (LVQ); the force-merge crosses it and trains LeanVec.
    @SneakyThrows
    public void testSVSVamana_withLeanVecDeferredTraining_forceMergeUpgrade_thenSucceed() {
        final String indexName = "test-svs-vamana-leanvec-deferred";
        final String fieldName = "test-field";
        final int dimension = 8;
        final int numDocs = 1200;
        final int threshold = 1000;

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, SVS_ENGINE)
            .startObject(PARAMETERS)
            .field("degree", 32)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, "leanvec")
            .startObject(PARAMETERS)
            .field("primary_bits", 4)
            .field("residual_bits", 8)
            .field("dimensions", 4)
            .field("training_threshold", threshold)
            .field("rough_training_threshold", threshold)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndexAssumingEncoderSupport(indexName, builder.toString());

        java.util.Random random = new java.util.Random(42);
        float[][] docs = new float[numDocs][dimension];
        for (int i = 0; i < numDocs; i++) {
            for (int d = 0; d < dimension; d++) {
                docs[i][d] = random.nextFloat();
            }
        }
        bulkAddKnnDocs(indexName, fieldName, docs, numDocs);
        refreshAllNonSystemIndices();
        assertEquals(numDocs, getDocCount(indexName));

        forceMergeKnnIndex(indexName, 1);

        int k = 10;
        Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, docs[0], k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
        assertEquals(k, results.size());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testSVSVamana_withSearchWindowSizeMethodParameter_thenSucceed() {
        final String indexName = "test-svs-vamana-sw-param";
        final String fieldName = "test-field";
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, SVS_ENGINE)
            .startObject(PARAMETERS)
            .field("degree", 64)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, "lvq")
            .startObject(PARAMETERS)
            .field("primary_bits", 4)
            .field("residual_bits", 4)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndexAssumingEncoderSupport(indexName, builder.toString());
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();
        assertEquals(DOCS.length, getDocCount(indexName));

        int k = 2;
        KNNQueryBuilder query = KNNQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(new float[] { 1.0f, 1.0f, 1.0f })
            .k(k)
            .methodParameters(Map.of("search_window_size", 64, "search_buffer_capacity", 96))
            .build();
        Response response = searchKNNIndex(indexName, query, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
        assertEquals(k, results.size());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testSVSVamana_withCosinesimil_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-cosine", SpaceType.COSINESIMIL, builder -> {
            builder.startObject(PARAMETERS).field("degree", 64).endObject();
        });
    }

    @FunctionalInterface
    private interface ParamsWriter {
        void write(XContentBuilder builder) throws Exception;
    }

    private void runIndexSearchRoundtrip(String indexName, SpaceType spaceType, ParamsWriter paramsWriter) throws Exception {
        final String fieldName = "test-field";
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, SVS_ENGINE);
        paramsWriter.write(builder);
        builder.endObject().endObject().endObject().endObject();

        createKnnIndexAssumingEncoderSupport(indexName, builder.toString());
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();
        assertEquals(DOCS.length, getDocCount(indexName));

        int k = 2;
        Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, new float[] { 1.0f, 1.0f, 1.0f }, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
        assertEquals(k, results.size());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testSVSVamana_whenModeOnDisk_thenRejected() {
        final String fieldName = "test-field";
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "on_disk")
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, SVS_ENGINE)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        ResponseException e = expectThrows(ResponseException.class, () -> createKnnIndex("test-svs-vamana-ondisk", builder.toString()));
        assertTrue(EntityUtils.toString(e.getResponse().getEntity()).contains("on_disk is not supported with svs_vamana"));
    }

    // ------------------------------------------------------------------ radial search

    // Query (1,1,1): squared L2 distances are doc "0" -> 0, "1" -> 3, "2" -> 12.
    private static final float[] RADIAL_QUERY = { 1.0f, 1.0f, 1.0f };

    @SneakyThrows
    public void testSVSVamana_withRadialMaxDistance_thenExactResults() {
        final String indexName = "test-svs-vamana-radial-dist";
        final String fieldName = "test-field";
        createSvsIndex(indexName, fieldName, SpaceType.L2, builder -> { builder.startObject(PARAMETERS).field("degree", 64).endObject(); });
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();

        KNNQueryBuilder query = KNNQueryBuilder.builder().fieldName(fieldName).vector(RADIAL_QUERY).maxDistance(5.0f).build();
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchKNNIndex(indexName, query, 10).getEntity()), fieldName);
        assertDocIds(results, "0", "1");

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testSVSVamana_withRadialMinScore_thenExactResults() {
        final String indexName = "test-svs-vamana-radial-score";
        final String fieldName = "test-field";
        createSvsIndex(indexName, fieldName, SpaceType.L2, builder -> { builder.startObject(PARAMETERS).field("degree", 64).endObject(); });
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();

        // L2 score = 1/(1+distance): doc "0" -> 1.0, "1" -> 0.25, "2" -> ~0.077.
        KNNQueryBuilder query = KNNQueryBuilder.builder().fieldName(fieldName).vector(RADIAL_QUERY).minScore(0.2f).build();
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchKNNIndex(indexName, query, 10).getEntity()), fieldName);
        assertDocIds(results, "0", "1");

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testSVSVamana_withRadialAndSearchWindowSize_thenSucceed() {
        final String indexName = "test-svs-vamana-radial-sw";
        final String fieldName = "test-field";
        createSvsIndex(indexName, fieldName, SpaceType.L2, builder -> { builder.startObject(PARAMETERS).field("degree", 64).endObject(); });
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();

        KNNQueryBuilder query = KNNQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(RADIAL_QUERY)
            .maxDistance(5.0f)
            .methodParameters(Map.of("search_window_size", 64, "search_buffer_capacity", 96))
            .build();
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchKNNIndex(indexName, query, 10).getEntity()), fieldName);
        assertDocIds(results, "0", "1");

        deleteKNNIndex(indexName);
    }

    // cosine min_score 0.3 converts to a negative radius: rejected at query build (400).
    @SneakyThrows
    public void testSVSVamana_whenRadialThresholdNonPositive_thenRejected() {
        final String indexName = "test-svs-vamana-radial-nonpos";
        final String fieldName = "test-field";
        createSvsIndex(
            indexName,
            fieldName,
            SpaceType.COSINESIMIL,
            builder -> { builder.startObject(PARAMETERS).field("degree", 64).endObject(); }
        );
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();

        KNNQueryBuilder query = KNNQueryBuilder.builder().fieldName(fieldName).vector(RADIAL_QUERY).minScore(0.3f).build();
        ResponseException e = expectThrows(ResponseException.class, () -> searchKNNIndex(indexName, query, 10));
        assertEquals(400, e.getResponse().getStatusLine().getStatusCode());
        assertTrue(EntityUtils.toString(e.getResponse().getEntity()).contains("non-positive radius"));

        deleteKNNIndex(indexName);
    }

    // ------------------------------------------------------------------ filtered search

    // 30 docs at (i,i,i), even i tagged "a", odd i tagged "b". filtered_exact_search_threshold=0 keeps the
    // filtered queries on the native ANN path (default settings would fall back to exact search).
    @SneakyThrows
    private void createFilterCorpus(String indexName, String fieldName) {
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.knn.advanced.filtered_exact_search_threshold", 0)
            .build();
        createKnnIndex(indexName, settings, svsMapping(fieldName, SpaceType.L2, builder -> {
            builder.startObject(PARAMETERS).field("degree", 64).endObject();
        }));
        for (int i = 0; i < 30; i++) {
            float v = i;
            addKnnDocWithAttributes(
                indexName,
                String.valueOf(i),
                fieldName,
                new float[] { v, v, v },
                Map.of("tag", i % 2 == 0 ? "a" : "b")
            );
        }
        refreshAllNonSystemIndices();
    }

    @SneakyThrows
    public void testSVSVamana_withFilteredKnn_thenExactResults() {
        final String indexName = "test-svs-vamana-filter-knn";
        final String fieldName = "test-field";
        createFilterCorpus(indexName, fieldName);

        // Nearest "a"-tagged docs to (0,0,0) are 0, 2, 4.
        KNNQueryBuilder query = KNNQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(new float[] { 0.0f, 0.0f, 0.0f })
            .k(3)
            .filter(QueryBuilders.termQuery("tag", "a"))
            .build();
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchKNNIndex(indexName, query, 3).getEntity()), fieldName);
        assertDocIds(results, "0", "2", "4");

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testSVSVamana_withFilteredRadial_thenExactResults() {
        final String indexName = "test-svs-vamana-filter-radial";
        final String fieldName = "test-field";
        createFilterCorpus(indexName, fieldName);

        // Distance from (0,0,0) to (i,i,i) is 3i^2; max_distance 30 admits i in {0,1,2,3}; tag "a" keeps {0,2}.
        KNNQueryBuilder query = KNNQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(new float[] { 0.0f, 0.0f, 0.0f })
            .maxDistance(30.0f)
            .filter(QueryBuilders.termQuery("tag", "a"))
            .build();
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchKNNIndex(indexName, query, 10).getEntity()), fieldName);
        assertDocIds(results, "0", "2");

        deleteKNNIndex(indexName);
    }

    // ------------------------------------------------------------------ nested (pinned rejection)

    @SneakyThrows
    public void testSVSVamana_whenNestedField_thenRejected() {
        final String indexName = "test-svs-vamana-nested";
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject("nested_field")
            .field("type", "nested")
            .startObject("properties")
            .startObject("v")
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, SVS_ENGINE)
            .startObject(PARAMETERS)
            .field("degree", 64)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());

        Request doc = new Request("POST", "/" + indexName + "/_doc/1?refresh=true");
        doc.setJsonEntity("{\"nested_field\": [{\"v\": [1.0, 1.0, 1.0]}, {\"v\": [2.0, 2.0, 2.0]}]}");
        client().performRequest(doc);

        Request search = new Request("GET", "/" + indexName + "/_search");
        search.setJsonEntity(
            "{\"query\": {\"nested\": {\"path\": \"nested_field\", "
                + "\"query\": {\"knn\": {\"nested_field.v\": {\"vector\": [1.0, 1.0, 1.0], \"k\": 2}}}}}}"
        );
        ResponseException e = expectThrows(ResponseException.class, () -> client().performRequest(search));
        assertTrue(EntityUtils.toString(e.getResponse().getEntity()).contains("Nested fields are not supported"));

        deleteKNNIndex(indexName);
    }

    // ------------------------------------------------------------------ helpers

    private void assertDocIds(List<KNNResult> results, String... expectedDocIds) {
        assertEquals(Set.of(expectedDocIds), results.stream().map(KNNResult::getDocId).collect(java.util.stream.Collectors.toSet()));
    }

    private String svsMapping(String fieldName, SpaceType spaceType, ParamsWriter paramsWriter) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, SVS_ENGINE);
        paramsWriter.write(builder);
        builder.endObject().endObject().endObject().endObject();
        return builder.toString();
    }

    private void createSvsIndex(String indexName, String fieldName, SpaceType spaceType, ParamsWriter paramsWriter) throws Exception {
        createKnnIndexAssumingEncoderSupport(indexName, svsMapping(fieldName, spaceType, paramsWriter));
    }

    /**
     * Creates the index, converting the LVQ "requires Intel SIMD" rejection into a JUnit assumption so the
     * LVQ suites skip (rather than fail) on hosts without AVX-512.
     */
    private void createKnnIndexAssumingEncoderSupport(String indexName, String mapping) throws Exception {
        try {
            createKnnIndex(indexName, mapping);
        } catch (ResponseException e) {
            String body = EntityUtils.toString(e.getResponse().getEntity());
            assumeFalse("LVQ compression requires Intel AVX-512; skipping on this host", body.contains("requires Intel SIMD support"));
            throw e;
        }
    }
}
