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
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Log4j2
public class HalfFloatIndexIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "half_float_test_index";
    private static final int DIMENSION = 4;
    /** Every component is exactly representable in half precision, so FP16 rounding is the identity. */
    private static final float[] EXACT_VECTOR = { 1.0f, 2.0f, 3.0f, 4.0f };
    /** 0.1 has no exact half-precision representation, so it must come back rounded. */
    private static final float[] INEXACT_VECTOR = { 0.1f, 0.1f, 0.1f, 0.1f };

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
    // SQ 1-bit (16x compression) — flat method, FP16 rescoring copy instead of FP32
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFlatSq1BitIndex_indexAndSearch() {
        String mapping = buildHalfFloatSq1BitMapping("l2");
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
    public void testHalfFloatFlatSq1BitIndex_innerProductSpace() {
        String mapping = buildHalfFloatSq1BitMapping("innerproduct");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 0.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.0f, 1.0f, 0.0f, 0.0f });

        float[] queryVector = { 1.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 2, queryVector, null), 2);
        String responseBody = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(responseBody));
    }

    @SneakyThrows
    public void testHalfFloatFlatSq1BitIndex_cosineSpace() {
        String mapping = buildHalfFloatSq1BitMapping("cosinesimil");
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

    @SneakyThrows
    public void testHalfFloatFlatSq1BitIndex_forceMerge() {
        String mapping = buildHalfFloatSq1BitMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        // Index docs one by one to create multiple segments
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);

        // Force merge into 1 segment — recomputes .veq from the merged (FP16-decoded) source
        forceMergeKnnIndex(INDEX_NAME, 1);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatFlatSq1BitIndex_closeAndReopen() {
        String mapping = buildHalfFloatSq1BitMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);
        forceMergeKnnIndex(INDEX_NAME, 1);

        closeKNNIndex(INDEX_NAME);
        openIndex(INDEX_NAME);
        ensureGreen(INDEX_NAME);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatFlatSq1BitIndex_indexSortedForceMerge() {
        final String sortFieldName = "sort_key";

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.sort.field", sortFieldName)
            .put("index.sort.order", "desc")
            .build();
        String mapping = buildHalfFloatSq1BitMappingWithSortField(sortFieldName);
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

    @SneakyThrows
    public void testHalfFloatFlatSq1BitIndex_deleteAndSearch() {
        String mapping = buildHalfFloatSq1BitMapping("l2");
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
        assertTrue(results.stream().noneMatch(r -> "2".equals(r.getDocId())));
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatFlatSq1BitIndex_multipleSegments() {
        String mapping = buildHalfFloatSq1BitMapping("l2");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.refresh_interval", "-1")
            .build();
        createKnnIndex(INDEX_NAME, settings, mapping);

        int totalDocs = 20;
        for (int i = 0; i < totalDocs; i++) {
            Float[] vec = { (float) i, (float) (i + 1), (float) (i + 2), (float) (i + 3) };
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, vec);
            if (i % 5 == 4) {
                flushIndex(INDEX_NAME, true);
            }
        }
        refreshIndex(INDEX_NAME);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 5, queryVector, null), 5);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(5, results.size());
        assertEquals("0", results.get(0).getDocId());
    }

    // Any compression level other than 1x (default) or 16x (this feature) must still be rejected.
    @SneakyThrows
    public void testHalfFloatFlat_withUnsupportedCompression_shouldFail() {
        // half_float supports only 1x and 16x - every level defined against FLOAT's 32 bits is rejected.
        for (String compression : new String[] { "2x", "4x", "8x", "32x" }) {
            String mapping = KNNJsonIndexMappingsBuilder.builder()
                .fieldName(FIELD_NAME)
                .dimension(DIMENSION)
                .vectorDataType("half_float")
                .compressionLevel(compression)
                .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("flat").spaceType("l2").build())
                .build()
                .getIndexMapping();

            // A distinct index per case, so a create that unexpectedly succeeds cannot make the next
            // iteration fail with "already exists" and hide which level was accepted.
            final String indexName = INDEX_NAME + "_" + compression;
            ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(indexName, mapping));
            assertTrue(compression + " -> " + ex.getMessage(), ex.getMessage().contains("compression"));
        }
    }

    // ────────────────────────────────────────────────────────────────────────────
    // SQ 1-bit (16x compression) — hnsw method, FP16 rescoring copy instead of FP32
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatHnswSq1BitIndex_indexAndSearch() {
        String mapping = buildHalfFloatHnswSq1BitMapping("l2");
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
    public void testHalfFloatHnswSq1BitIndex_innerProductSpace() {
        String mapping = buildHalfFloatHnswSq1BitMapping("innerproduct");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 0.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.0f, 1.0f, 0.0f, 0.0f });

        float[] queryVector = { 1.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 2, queryVector, null), 2);
        String responseBody = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(responseBody));
    }

    @SneakyThrows
    public void testHalfFloatHnswSq1BitIndex_cosineSpace() {
        String mapping = buildHalfFloatHnswSq1BitMapping("cosinesimil");
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

    @SneakyThrows
    public void testHalfFloatHnswSq1BitIndex_forceMerge() {
        String mapping = buildHalfFloatHnswSq1BitMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        // Index docs one by one to create multiple segments, each with its own HNSW graph
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);

        // Force merge into 1 segment — rebuilds the HNSW graph and recomputes .veq from the merged
        // (FP16-decoded) source
        forceMergeKnnIndex(INDEX_NAME, 1);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatHnswSq1BitIndex_closeAndReopen() {
        String mapping = buildHalfFloatHnswSq1BitMapping("l2");
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);
        forceMergeKnnIndex(INDEX_NAME, 1);

        closeKNNIndex(INDEX_NAME);
        openIndex(INDEX_NAME);
        ensureGreen(INDEX_NAME);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatHnswSq1BitIndex_indexSortedForceMerge() {
        final String sortFieldName = "sort_key";

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.sort.field", sortFieldName)
            .put("index.sort.order", "desc")
            .build();
        String mapping = buildHalfFloatHnswSq1BitMappingWithSortField(sortFieldName);
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

    @SneakyThrows
    public void testHalfFloatHnswSq1BitIndex_deleteAndSearch() {
        String mapping = buildHalfFloatHnswSq1BitMapping("l2");
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
        assertTrue(results.stream().noneMatch(r -> "2".equals(r.getDocId())));
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatHnswSq1BitIndex_multipleSegments() {
        String mapping = buildHalfFloatHnswSq1BitMapping("l2");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.refresh_interval", "-1")
            .build();
        createKnnIndex(INDEX_NAME, settings, mapping);

        int totalDocs = 20;
        for (int i = 0; i < totalDocs; i++) {
            Float[] vec = { (float) i, (float) (i + 1), (float) (i + 2), (float) (i + 3) };
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, vec);
            if (i % 5 == 4) {
                flushIndex(INDEX_NAME, true);
            }
        }
        refreshIndex(INDEX_NAME);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 5, queryVector, null), 5);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(5, results.size());
        assertEquals("0", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatHnsw_withExplicitEncoder_shouldFail() {
        String mapping = buildHalfFloatHnswSqMapping("l2", 7);
        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping));
        assertTrue(ex.getMessage(), ex.getMessage().contains("encoder"));
        assertTrue(ex.getMessage(), ex.getMessage().contains("compression_level"));
    }

    @SneakyThrows
    public void testHalfFloatHnsw_withUnsupportedCompression_shouldFail() {
        // half_float supports only 1x and 16x - every level defined against FLOAT's 32 bits is rejected.
        for (String compression : new String[] { "2x", "4x", "8x", "32x" }) {
            String mapping = KNNJsonIndexMappingsBuilder.builder()
                .fieldName(FIELD_NAME)
                .dimension(DIMENSION)
                .vectorDataType("half_float")
                .compressionLevel(compression)
                .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("hnsw").engine("lucene").spaceType("l2").build())
                .build()
                .getIndexMapping();

            final String indexName = INDEX_NAME + "_" + compression;
            ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(indexName, mapping));
            assertTrue(compression + " -> " + ex.getMessage(), ex.getMessage().contains("compression"));
        }
    }

    @SneakyThrows
    public void testHalfFloatHnswOnDisk_indexAndSearch() {
        // ON_DISK now resolves half_float to x16 (SQ 1-bit), the counterpart of FLOAT's ON_DISK -> x32.
        String mapping = "{"
            + "\"properties\":{"
            + "\""
            + FIELD_NAME
            + "\":{"
            + "\"type\":\"knn_vector\","
            + "\"dimension\":"
            + DIMENSION
            + ","
            + "\"data_type\":\"half_float\","
            + "\"mode\":\"on_disk\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"lucene\","
            + "\"space_type\":\"l2\""
            + "}}}}";
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(3, results.size());
    }

    @SneakyThrows
    public void testHalfFloatHnswOnDiskWithX1_shouldFail() {
        String mapping = "{"
            + "\"properties\":{"
            + "\""
            + FIELD_NAME
            + "\":{"
            + "\"type\":\"knn_vector\","
            + "\"dimension\":"
            + DIMENSION
            + ","
            + "\"data_type\":\"half_float\","
            + "\"mode\":\"on_disk\","
            + "\"compression_level\":\"1x\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"lucene\","
            + "\"space_type\":\"l2\""
            + "}}}}";

        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping));
        assertTrue(ex.getMessage(), ex.getMessage().contains("x1"));
        assertTrue(ex.getMessage(), ex.getMessage().contains("on_disk"));
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
    // Binary DocValues (index.knn=false). No ANN structure is built, so the supported
    // access paths are script scoring, docvalue_fields and _source - same as every other
    // data type on this path.
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatWithoutMethod_indexKnnFalse_scriptScore() {
        createHalfFloatFlatIndex();

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        refreshIndex(INDEX_NAME);

        String query = buildKnnScoreScriptQuery(new float[] { 0.0f, 0.0f, 0.0f, 0.0f });
        Response response = searchKNNIndex(INDEX_NAME, query, 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(3, results.size());
        assertEquals("3", results.get(0).getDocId());
        assertEquals("1", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());
    }

    /**
     * The DocValues bytes are FP16, so a reader that resolved the data type incorrectly would decode them as FP32
     * and hand back a vector of the wrong length. Reading the vectors back through the script doc values catches that.
     */
    @SneakyThrows
    public void testHalfFloatWithoutMethod_indexKnnFalse_vectorRoundTripsThroughDocValues() {
        createHalfFloatFlatIndex();

        Float[] vector = { 1.5f, -2.25f, 3.125f, 4.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);
        refreshIndex(INDEX_NAME);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("script_score")
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("script")
            .field("source", "doc['" + FIELD_NAME + "'].value.length")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = searchKNNIndex(INDEX_NAME, query, 1);
        List<Float> scores = parseSearchResponseScore(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(1, scores.size());
        assertEquals("Vector read back from DocValues should have the mapped dimension", (float) DIMENSION, scores.get(0), 0.0f);
    }

    /**
     * Binary DocValues are merged by the default Lucene codec here, since no k-NN codec is installed when
     * index.knn is false. Force merging exercises that path for the FP16 encoding.
     */
    @SneakyThrows
    public void testHalfFloatWithoutMethod_indexKnnFalse_forceMerge() {
        createHalfFloatFlatIndex();

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);

        forceMergeKnnIndex(INDEX_NAME, 1);

        String query = buildKnnScoreScriptQuery(new float[] { 0.0f, 0.0f, 0.0f, 0.0f });
        Response response = searchKNNIndex(INDEX_NAME, query, 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(3, results.size());
        assertEquals("3", results.get(0).getDocId());
        assertEquals("1", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatWithoutMethod_indexKnnFalse_rejectsOutOfRangeValue() {
        createHalfFloatFlatIndex();

        // 70000 exceeds the half_float maximum (65504), so it must be rejected rather than silently stored as Inf
        ResponseException ex = expectThrows(
            ResponseException.class,
            () -> addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 70000.0f, 1.0f, 2.0f, 3.0f })
        );
        assertTrue(ex.getMessage(), ex.getMessage().contains("half_float"));
    }

    private void createHalfFloatFlatIndex() throws IOException {
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .build()
            .getIndexMapping();

        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", false).build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private String buildKnnScoreScriptQuery(final float[] queryVector) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("script_score")
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("script")
            .field("source", "knn_score")
            .field("lang", "knn")
            .startObject("params")
            .field("field", FIELD_NAME)
            .field("query_value", queryVector)
            .field("space_type", SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────────────

    // ────────────────────────────────────────────────────────────────────────────
    // HNSW (Faiss engine) - x1 stores FP16 as-is, x16 is SQ 1-bit. Faiss has no flat method for
    // half_float and no IVF support, so HNSW is the whole surface.
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFaissHnswIndex_indexAndSearch() {
        createKnnIndex(INDEX_NAME, buildHalfFloatFaissHnswMapping("l2", null));

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(3, results.size());
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatFaissHnswIndex_explicitX1() {
        createKnnIndex(INDEX_NAME, buildHalfFloatFaissHnswMapping("l2", "1x"));

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 2, queryVector, null), 2);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(2, results.size());
        assertEquals("2", results.get(0).getDocId());
    }

    /** Merge is where the FP16 vectors get rewritten, so exercise it rather than trusting flush alone. */
    @SneakyThrows
    public void testHalfFloatFaissHnswSq1BitIndex_forceMerge() {
        createKnnIndex(INDEX_NAME, buildHalfFloatFaissHnswMapping("l2", "16x"));

        for (int i = 1; i <= 6; i++) {
            addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, new Float[] { (float) i, (float) i, (float) i, (float) i });
            refreshIndex(INDEX_NAME);
        }
        forceMergeKnnIndex(INDEX_NAME, 1);

        float[] queryVector = { 1.0f, 1.0f, 1.0f, 1.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 6, queryVector, null), 6);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(6, results.size());
    }

    @SneakyThrows
    public void testHalfFloatFaissHnswOnDisk_indexAndSearch() {
        // ON_DISK resolves half_float to x16 (SQ 1-bit), the counterpart of FLOAT's ON_DISK -> x32.
        createKnnIndex(INDEX_NAME, buildHalfFloatFaissHnswOnDiskMapping("l2", null));

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(3, results.size());
    }

    // Faiss FLOAT accepts ON_DISK + 1x, so half_float does too - matching its own engine, not Lucene's.
    @SneakyThrows
    public void testHalfFloatFaissHnswOnDiskWithX1_indexAndSearch() {
        createKnnIndex(INDEX_NAME, buildHalfFloatFaissHnswOnDiskMapping("l2", "1x"));

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 2, queryVector, null), 2);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(2, results.size());
        assertEquals("2", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatFaissHnswOnDisk_withUnsupportedCompression_shouldFail() {
        for (String compression : new String[] { "2x", "4x", "8x", "32x" }) {
            final String indexName = INDEX_NAME + "_ondisk_" + compression;
            ResponseException ex = expectThrows(
                ResponseException.class,
                () -> createKnnIndex(indexName, buildHalfFloatFaissHnswOnDiskMapping("l2", compression))
            );
            assertTrue(compression + " -> " + ex.getMessage(), ex.getMessage().contains("compression"));
        }
    }

    private String buildHalfFloatFaissHnswOnDiskMapping(String spaceType, String compressionLevel) {
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
            + "\"mode\":\"on_disk\","
            + (compressionLevel == null ? "" : "\"compression_level\":\"" + compressionLevel + "\",")
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"faiss\","
            + "\"space_type\":\""
            + spaceType
            + "\""
            + "}}}}";
    }

    @SneakyThrows
    public void testHalfFloatFaissHnsw_withUnsupportedCompression_shouldFail() {
        for (String compression : new String[] { "2x", "4x", "8x", "32x" }) {
            // A distinct index per case: a create that unexpectedly succeeds would otherwise make the
            // next iteration fail with "already exists" and hide which level was actually accepted.
            final String indexName = INDEX_NAME + "_" + compression;
            ResponseException ex = expectThrows(
                ResponseException.class,
                () -> createKnnIndex(indexName, buildHalfFloatFaissHnswMapping("l2", compression))
            );
            assertTrue(compression + " -> " + ex.getMessage(), ex.getMessage().contains("compression"));
        }
    }

    @SneakyThrows
    public void testHalfFloatFaissHnsw_withExplicitEncoder_shouldFail() {
        ResponseException ex = expectThrows(
            ResponseException.class,
            () -> createKnnIndex(INDEX_NAME + "_encoder", buildHalfFloatFaissHnswSqMapping("l2", 1))
        );
        assertTrue(ex.getMessage(), ex.getMessage().contains("encoder"));
        assertTrue(ex.getMessage(), ex.getMessage().contains("compression_level"));
    }

    @SneakyThrows
    public void testHalfFloatFaissIvf_shouldFail() {
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("ivf").engine("faiss").spaceType("l2").build())
            .build()
            .getIndexMapping();

        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping));
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Filtered search and vector retrieval
    // ────────────────────────────────────────────────────────────────────────────

    @SneakyThrows
    public void testHalfFloatFlatIndex_filteredSearch() {
        String filterFieldName = "parking";
        createKnnIndex(INDEX_NAME, buildHalfFloatMappingWithKeywordField(filterFieldName));

        // Docs 1 and 3 match the filter, doc 2 does not. Doc 3 is nearest to the query vector.
        addKnnDocWithAttributes(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 1.0f, 1.0f, 1.0f }, Map.of(filterFieldName, "true"));
        addKnnDocWithAttributes(INDEX_NAME, "2", FIELD_NAME, new Float[] { 2.0f, 2.0f, 2.0f, 2.0f }, Map.of(filterFieldName, "false"));
        addKnnDocWithAttributes(INDEX_NAME, "3", FIELD_NAME, new Float[] { 3.0f, 3.0f, 3.0f, 3.0f }, Map.of(filterFieldName, "true"));
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        float[] queryVector = { 3.0f, 3.0f, 3.0f, 3.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildFilteredSearchQuery(queryVector, 3, filterFieldName, "true"), 3);
        String entity = EntityUtils.toString(response.getEntity());

        List<String> docIds = parseIds(entity);
        assertEquals("Filter should exclude doc 2", 2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));
    }

    @SneakyThrows
    public void testHalfFloatFlatIndex_docValueFields() {
        createKnnIndex(INDEX_NAME, buildHalfFloatMapping("l2"));
        indexFp16ExactAndInexactDocs();

        Response response = searchKNNIndex(INDEX_NAME, buildDocValueFieldsQuery(), 10);
        Map<String, List<Double>> byDocId = parseDocValueVectors(EntityUtils.toString(response.getEntity()));

        assertEquals(2, byDocId.size());
        assertFp16Vector("doc 1 (exactly representable)", byDocId.get("1"), EXACT_VECTOR);
        assertFp16Vector("doc 2 (not representable)", byDocId.get("2"), INEXACT_VECTOR);
    }

    @SneakyThrows
    public void testHalfFloatFlatIndex_knnDerivedSource() {
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.knn.derived_source.enabled", true)
            .build();
        createKnnIndex(INDEX_NAME, settings, buildHalfFloatMapping("l2"));
        indexFp16ExactAndInexactDocs();

        Response response = searchKNNIndex(INDEX_NAME, buildMatchAllSourceQuery(), 10);
        Map<String, List<Double>> byDocId = parseSourceVectors(EntityUtils.toString(response.getEntity()));

        assertEquals(2, byDocId.size());
        assertFp16Vector("doc 1 (exactly representable)", byDocId.get("1"), EXACT_VECTOR);
        assertFp16Vector("doc 2 (not representable)", byDocId.get("2"), INEXACT_VECTOR);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Helpers - mappings
    // ────────────────────────────────────────────────────────────────────────────

    private String buildHalfFloatFaissHnswMapping(String spaceType, String compressionLevel) throws Exception {
        KNNJsonIndexMappingsBuilder.KNNJsonIndexMappingsBuilderBuilder builder = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("hnsw").engine("faiss").spaceType(spaceType).build());
        if (compressionLevel != null) {
            builder.compressionLevel(compressionLevel);
        }
        return builder.build().getIndexMapping();
    }

    private String buildHalfFloatFaissHnswSqMapping(String spaceType, int bits) {
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
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"faiss\","
            + "\"space_type\":\""
            + spaceType
            + "\","
            + "\"parameters\":{\"encoder\":{\"name\":\"sq\",\"parameters\":{\"bits\":"
            + bits
            + "}}}"
            + "}}}}";
    }

    private String buildHalfFloatMapping(String spaceType) throws Exception {
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("flat").spaceType(spaceType).build())
            .build()
            .getIndexMapping();
    }

    private String buildHalfFloatSq1BitMapping(String spaceType) throws Exception {
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .compressionLevel("16x")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("flat").spaceType(spaceType).build())
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

    private String buildHalfFloatHnswSq1BitMapping(String spaceType) {
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
            + "\"compression_level\":\"16x\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"lucene\","
            + "\"space_type\":\""
            + spaceType
            + "\""
            + "}"
            + "}"
            + "}}";
    }

    private String buildHalfFloatHnswSqMapping(String spaceType, int bits) {
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
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"lucene\","
            + "\"space_type\":\""
            + spaceType
            + "\","
            + "\"parameters\":{"
            + "\"encoder\":{\"name\":\"sq\",\"parameters\":{\"bits\":"
            + bits
            + "}}"
            + "}"
            + "}"
            + "}"
            + "}}";
    }

    private String buildHalfFloatHnswSq1BitMappingWithSortField(String sortFieldName) {
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
            + "\"compression_level\":\"16x\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"lucene\","
            + "\"space_type\":\"l2\""
            + "}"
            + "},"
            + "\""
            + sortFieldName
            + "\":{\"type\":\"long\"}"
            + "}}";
    }

    private String buildHalfFloatMappingWithSortField(String sortFieldName) {
        return buildHalfFloatMappingWithExtraField(sortFieldName, "long");
    }

    private String buildHalfFloatMappingWithKeywordField(String keywordFieldName) {
        return buildHalfFloatMappingWithExtraField(keywordFieldName, "keyword");
    }

    private String buildHalfFloatMappingWithExtraField(String extraFieldName, String extraFieldType) {
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
            + "\"method\":{\"name\":\"flat\",\"space_type\":\"l2\"}"
            + "},"
            + "\""
            + extraFieldName
            + "\":{\"type\":\""
            + extraFieldType
            + "\"}"
            + "}}";
    }

    private String buildHalfFloatSq1BitMappingWithSortField(String sortFieldName) {
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
            + "\"compression_level\":\"16x\","
            + "\"method\":{\"name\":\"flat\",\"space_type\":\"l2\"}"
            + "},"
            + "\""
            + sortFieldName
            + "\":{\"type\":\"long\"}"
            + "}}";
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Helpers - queries
    // ────────────────────────────────────────────────────────────────────────────

    private String buildFilteredSearchQuery(float[] queryVector, int k, String filterFieldName, String filterValue) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field("k", k)
            .startObject("filter")
            .startObject("term")
            .field(filterFieldName, filterValue)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    private String buildDocValueFieldsQuery() throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startArray("docvalue_fields")
            .startObject()
            .field("field", FIELD_NAME)
            .field("format", "array")
            .endObject()
            .endArray()
            .field("_source", false)
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();
    }

    private String buildMatchAllSourceQuery() throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Helpers - FP16 round-trip assertions
    // ────────────────────────────────────────────────────────────────────────────

    private void indexFp16ExactAndInexactDocs() throws Exception {
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.1f, 0.1f, 0.1f, 0.1f });
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);
    }

    private void assertFp16Vector(String context, List<Double> actual, float[] indexedValues) {
        assertNotNull(context + ": vector should be present", actual);
        assertEquals(context + ": dimension mismatch", indexedValues.length, actual.size());
        for (int i = 0; i < indexedValues.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(indexedValues[i]));
            assertEquals(context + ": value " + i, expected, actual.get(i).floatValue(), 0.0f);
        }
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Helpers - response parsing
    // ────────────────────────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> parseHitsList(String responseBody) throws IOException {
        Map<String, Object> responseMap = createParser(org.opensearch.common.xcontent.json.JsonXContent.jsonXContent, responseBody).map();
        return (List<Map<String, Object>>) ((Map<String, Object>) responseMap.get("hits")).get("hits");
    }

    @SuppressWarnings("unchecked")
    private Map<String, List<Double>> parseDocValueVectors(String responseBody) throws IOException {
        Map<String, List<Double>> byDocId = new HashMap<>();
        for (Map<String, Object> hit : parseHitsList(responseBody)) {
            assertNull("_source should be disabled", hit.get("_source"));
            Map<String, Object> fields = (Map<String, Object>) hit.get("fields");
            assertNotNull("fields should be present", fields);
            List<List<Double>> vectorField = (List<List<Double>>) fields.get(FIELD_NAME);
            assertNotNull("docvalue_fields should return the vector", vectorField);
            assertFalse("docvalue_fields vector should not be empty", vectorField.isEmpty());
            byDocId.put((String) hit.get("_id"), vectorField.get(0));
        }
        return byDocId;
    }

    @SuppressWarnings("unchecked")
    private Map<String, List<Double>> parseSourceVectors(String responseBody) throws IOException {
        Map<String, List<Double>> byDocId = new HashMap<>();
        for (Map<String, Object> hit : parseHitsList(responseBody)) {
            Map<String, Object> source = (Map<String, Object>) hit.get("_source");
            assertNotNull("_source should be reconstructed", source);
            List<Double> vector = (List<Double>) source.get(FIELD_NAME);
            assertNotNull("_source should contain the vector field", vector);
            byDocId.put((String) hit.get("_id"), vector);
        }
        return byDocId;
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Helpers - index operations
    // ────────────────────────────────────────────────────────────────────────────

    private void flushIndex(String index, boolean force) throws Exception {
        Request request = new Request("POST", "/" + index + "/_flush");
        request.addParameter("force", String.valueOf(force));
        client().performRequest(request);
    }

    @SneakyThrows
    public void testHalfFloatWithFaissFlat_memoryOptimizedSearchEnabled_indexAndSearch() {
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("hnsw").engine("faiss").spaceType("l2").build())
            .build()
            .getIndexMapping();
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true)
            .build();
        createKnnIndex(INDEX_NAME, settings, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        assertEquals(3, parseHits(responseBody));
    }

    @SneakyThrows
    public void testHalfFloatWithFaissFlat_memoryOptimizedSearchEnabled_forceMergeThenReopen() {
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(KNNJsonIndexMappingsBuilder.Method.builder().methodName("hnsw").engine("faiss").spaceType("l2").build())
            .build()
            .getIndexMapping();
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true)
            .build();
        createKnnIndex(INDEX_NAME, settings, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);
        forceMergeKnnIndex(INDEX_NAME, 1);

        closeKNNIndex(INDEX_NAME);
        openIndex(INDEX_NAME);
        ensureGreen(INDEX_NAME);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatWithFaissFlat_forceMerge_thenSearch() {
        String mapping = "{"
            + "\"properties\":{"
            + "\""
            + FIELD_NAME
            + "\":{"
            + "\"type\":\"knn_vector\","
            + "\"dimension\":"
            + DIMENSION
            + ","
            + "\"data_type\":\"half_float\","
            + "\"compression_level\":\"1x\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"faiss\","
            + "\"space_type\":\"l2\""
            + "}"
            + "}"
            + "}}";
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

    @SneakyThrows
    public void testHalfFloatWithFaissSq16_shouldFail() {
        String mapping = "{"
            + "\"properties\":{"
            + "\""
            + FIELD_NAME
            + "\":{"
            + "\"type\":\"knn_vector\","
            + "\"dimension\":"
            + DIMENSION
            + ","
            + "\"data_type\":\"half_float\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"faiss\","
            + "\"space_type\":\"l2\","
            + "\"parameters\":{\"encoder\":{\"name\":\"sq\",\"parameters\":{\"bits\":16}}}"
            + "}"
            + "}"
            + "}}";

        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping));
        assertTrue(ex.getMessage().contains("half_float"));
    }

    @SneakyThrows
    public void testHalfFloatWithFaissSq1Bit_indexAndSearch() {
        String mapping = "{"
            + "\"properties\":{"
            + "\""
            + FIELD_NAME
            + "\":{"
            + "\"type\":\"knn_vector\","
            + "\"dimension\":"
            + DIMENSION
            + ","
            + "\"data_type\":\"half_float\","
            + "\"compression_level\":\"16x\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"faiss\","
            + "\"space_type\":\"l2\""
            + "}"
            + "}"
            + "}}";
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
    public void testHalfFloatWithFaissFlat_closeAndReopen() {
        String mapping = "{"
            + "\"properties\":{"
            + "\""
            + FIELD_NAME
            + "\":{"
            + "\"type\":\"knn_vector\","
            + "\"dimension\":"
            + DIMENSION
            + ","
            + "\"data_type\":\"half_float\","
            + "\"compression_level\":\"1x\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"faiss\","
            + "\"space_type\":\"l2\""
            + "}"
            + "}"
            + "}}";
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);
        forceMergeKnnIndex(INDEX_NAME, 1);

        closeKNNIndex(INDEX_NAME);
        openIndex(INDEX_NAME);
        ensureGreen(INDEX_NAME);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatWithFaissSq1Bit_closeAndReopen() {
        String mapping = "{"
            + "\"properties\":{"
            + "\""
            + FIELD_NAME
            + "\":{"
            + "\"type\":\"knn_vector\","
            + "\"dimension\":"
            + DIMENSION
            + ","
            + "\"data_type\":\"half_float\","
            + "\"compression_level\":\"16x\","
            + "\"method\":{"
            + "\"name\":\"hnsw\","
            + "\"engine\":\"faiss\","
            + "\"space_type\":\"l2\""
            + "}"
            + "}"
            + "}}";
        createKnnIndex(INDEX_NAME, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        flushIndex(INDEX_NAME, true);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });
        flushIndex(INDEX_NAME, true);
        forceMergeKnnIndex(INDEX_NAME, 1);

        closeKNNIndex(INDEX_NAME);
        openIndex(INDEX_NAME);
        ensureGreen(INDEX_NAME);

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(3, results.size());
        // Closest to origin should be doc 3
        assertEquals("3", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHalfFloatWithoutMethod_indexKnnTrue_thenResolvesToFaissFlat() {
        // HALF_FLOAT without an explicit method now succeeds: default engine is Faiss, and Faiss
        // HNSW with no encoder specified resolves to the flat encoder, which supports half_float.
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .build()
            .getIndexMapping();

        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", true).build();
        createKnnIndex(INDEX_NAME, settings, mapping);

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.1f, 0.2f, 0.3f, 0.4f });

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(FIELD_NAME, 3, queryVector, null), 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        assertEquals(3, parseHits(responseBody));
    }
}
