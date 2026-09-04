/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Log4j2
public class HalfFloatIndexIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "half_float_test_index";
    private static final int DIMENSION = 4;

    // ────────────────────────────────────────────────────────────────────────────
    // Basic indexing and search
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFlatIndex_indexAndSearch() {
        String mapping = buildHalfFloatMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        assertEquals(3, parseHits(responseBody));
    }

    @SneakyThrows
    public void testHalfFloatFlatIndex_innerProductSpace() {
        String mapping = buildHalfFloatMapping("innerproduct");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 0.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.0f, 1.0f, 0.0f, 0.0f });

        float[] queryVector = { 1.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 2, queryVector, null), 2);
        String responseBody = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(responseBody));
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Cosine space type
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFlatIndex_cosineSpace() {
        String mapping = buildHalfFloatMapping("cosinesimil");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 0.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.0f, 1.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 1.0f, 1.0f, 0.0f, 0.0f });

        float[] queryVector = { 1.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Doc 1 is identical direction to query, should be the top result
        assertEquals("1", results.get(0).getDocId());
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Force merge - exercises mergeOneFlatVectorField path
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFlatIndex_forceMerge() {
        String mapping = buildHalfFloatMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        // Index docs one by one to create multiple segments
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);

        // Force merge into 1 segment
        forceMergeKnnIndex(INDEX_NAME, 1);

        // Verify search still works after merge
        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatFlatIndex_indexSortedForceMerge() {
        final String sortFieldName = "sort_key";

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.sort.field", sortFieldName)
            .put("index.sort.order", "desc")
            .build();
        String mapping = buildHalfFloatMappingWithSortField(sortFieldName);
        createIndex(INDEX_NAME, settings, mapping.substring(1, mapping.length() - 1));

        Float[][] vectors = {
            { 1.0f, 2.0f, 3.0f, 4.0f },
            { 5.0f, 6.0f, 7.0f, 8.0f },
            { 0.1f, 0.2f, 0.3f, 0.4f },
            { 10.0f, 10.0f, 10.0f, 10.0f } };
        for (int i = 0; i < vectors.length; i++) {
            addKnnDocWithAttributes(INDEX_NAME, String.valueOf(i + 1), FIELD_NAME, vectors[i], Map.of(sortFieldName, String.valueOf(i)));
        }

        forceMergeKnnIndex(INDEX_NAME, 1);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 4, queryVector, null), 4);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(4, results.size());
        assertEquals("3", results.get(0).getDocId());
        assertEquals("1", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());
        assertEquals("4", results.get(3).getDocId());
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Delete + search - sparse segment exercises ordToDoc mapping
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFlatIndex_deleteAndSearch() {
        String mapping = buildHalfFloatMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, new Float[] { 10.0f, 10.0f, 10.0f, 10.0f });

        // Delete doc 2 (creates sparse segment)
        deleteKnnDoc(INDEX_NAME, "2");

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Deleted doc should not appear
        assertTrue(results.stream().noneMatch(r -> "2".equals(r.getDocId())));
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Script scoring with knn enabled
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFlatIndex_scriptScoreL2() {
        testHalfFloatScriptScore(SpaceType.L2);
    }

    @SneakyThrows
    public void testHalfFloatFlatIndex_scriptScoreCosine() {
        testHalfFloatScriptScore(SpaceType.COSINESIMIL);
    }

    @SneakyThrows
    public void testHalfFloatFlatIndex_scriptScoreInnerProduct() {
        testHalfFloatScriptScore(SpaceType.INNER_PRODUCT);
    }

    private void testHalfFloatScriptScore(SpaceType spaceType) throws Exception {
        String mapping = buildHalfFloatMapping(spaceType.getValue());
        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", true).build();
        createKnnIndex(INDEX_NAME, settings, mapping);

        try {
            addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 1.0f, 1.0f, 1.0f });
            addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 2.0f, 2.0f, 2.0f, 2.0f });
            addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 3.0f, 3.0f, 3.0f, 3.0f });

            float[] queryVector = { 1.0f, 1.0f, 1.0f, 1.0f };

            QueryBuilder qb = new MatchAllQueryBuilder();
            Map<String, Object> params = new HashMap<>();
            params.put("field", FIELD_NAME);
            params.put("query_value", queryVector);
            params.put("space_type", spaceType.getValue());

            Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params, 3);
            Response response = client().performRequest(request);
            assertEquals(200, response.getStatusLine().getStatusCode());

            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
            assertEquals(3, results.size());
            if (spaceType == SpaceType.COSINESIMIL) {
                // All vectors are parallel so any doc could be first; just verify we got results
                assertNotNull(results.get(0).getDocId());
            } else if (spaceType == SpaceType.INNER_PRODUCT) {
                // Inner product: highest dot product wins, doc 3 has largest magnitude
                assertEquals("3", results.get(0).getDocId());
            } else {
                // L2: smallest distance wins, doc 1 is identical to the query
                assertEquals("1", results.get(0).getDocId());
            }
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Multiple docs across segments - realistic scenario
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFlatIndex_multipleSegments() {
        String mapping = buildHalfFloatMapping("l2");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.refresh_interval", "-1")
            .build();
        createKnnIndex(INDEX_NAME, settings, mapping);

        // Create multiple segments with multiple docs each
        int totalDocs = 20;
        for (int i = 0; i < totalDocs; i++) {
            Float[] vec = { (float) i, (float) (i + 1), (float) (i + 2), (float) (i + 3) };
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, vec);
            if (i % 5 == 4) {
                flushIndex(INDEX_NAME, true);
            }
        }
        refreshIndex(INDEX_NAME);

        // Search for k=5 nearest to origin
        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 5, queryVector, null), 5);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(5, results.size());
        assertEquals("0", results.get(0).getDocId());
    }

    // ────────────────────────────────────────────────────────────────────────────
    // HNSW (Lucene engine)
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatHnswIndex_indexAndSearch() {
        String mapping = buildHalfFloatHnswMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatHnswIndex_innerProductSpace() {
        String mapping = buildHalfFloatHnswMapping("innerproduct");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 0.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.0f, 1.0f, 0.0f, 0.0f });

        float[] queryVector = { 1.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 2, queryVector, null), 2);
        String responseBody = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(responseBody));
    }

    @SneakyThrows
    public void testHalfFloatHnswIndex_forceMerge() {
        String mapping = buildHalfFloatHnswMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        // Index docs one by one to create multiple segments, each with its own HNSW graph
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);

        // Force merge into 1 segment - rebuilds a single HNSW graph from the merged flat storage
        forceMergeKnnIndex(INDEX_NAME, 1);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatHnswIndex_forceMerge_cosine() {
        String mapping = buildHalfFloatHnswMapping("cosinesimil");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.4f, 0.3f, 0.2f, 0.1f });
        flushIndex(INDEX_NAME, true);

        forceMergeKnnIndex(INDEX_NAME, 1);

        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("1", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatHnswIndex_forceMerge_innerProduct() {
        String mapping = buildHalfFloatHnswMapping("innerproduct");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 0.0f, 0.0f, 0.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.0f, 1.0f, 0.0f, 0.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.0f, 0.0f, 1.0f, 0.0f });
        flushIndex(INDEX_NAME, true);

        forceMergeKnnIndex(INDEX_NAME, 1);

        float[] queryVector = { 1.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("1", results.get(0).getDocId());
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Negative tests - blocked configurations
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatWithoutMethod_indexKnnFalse_shouldFail() {
        // HALF_FLOAT is not supported for the binary DocValues (index.knn=false) path
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .build()
            .getIndexMapping();

        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", false).build();

        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, settings, mapping));
        assertTrue(
            "Should reject half_float when index.knn is disabled",
            ex.getMessage().contains("HALF_FLOAT") || ex.getMessage().contains("half_float")
        );
    }

    @SneakyThrows
    public void testHalfFloatWithoutMethod_indexKnnTrue_shouldFail() {
        // half_float has no default method: default engine resolution picks Faiss, which rejects the
        // data type. An explicit lucene flat/hnsw method is required.
        // TODO: Remove this test once FP16 support lands for Faiss -- half_float will then resolve to
        // faiss hnsw by default, exactly like float, and this will no longer fail.
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .build()
            .getIndexMapping();

        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", true).build();

        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, settings, mapping));
        assertTrue(
            "Should reject half_float without explicit method=flat",
            ex.getMessage().contains("HALF_FLOAT") || ex.getMessage().contains("half_float") || ex.getMessage().contains("data_type")
        );
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────────────

    private String buildHalfFloatMapping(String spaceType) throws Exception {
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("flat").engine("lucene").spaceType(spaceType).build())
            .build()
            .getIndexMapping();
    }

    private String buildHalfFloatHnswMapping(String spaceType) throws Exception {
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("hnsw").engine("lucene").spaceType(spaceType).build())
            .build()
            .getIndexMapping();
    }

    private String buildHalfFloatMappingWithSortField(String sortFieldName) {
        return "{"
            + "\"properties\":{"
            + "\""
            + FIELD_NAME
            + "\":{"
            + "\"type\":\"knn_vector\","
            + "\"dimension\":"
            + DIMENSION
            + ","
            + "\"data_type\":\"half_float\","
            + "\"method\":{\"name\":\"flat\",\"engine\":\"lucene\",\"space_type\":\"l2\"}"
            + "},"
            + "\""
            + sortFieldName
            + "\":{\"type\":\"long\"}"
            + "}}";
    }

    private void flushIndex(String index, boolean force) throws Exception {
        Request request = new Request("POST", "/" + index + "/_flush");
        request.addParameter("force", String.valueOf(force));
        client().performRequest(request);
    }
}
