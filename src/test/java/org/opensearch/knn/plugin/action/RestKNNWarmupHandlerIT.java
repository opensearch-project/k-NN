/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.action;

import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.knn.KNNRestTestCase;
import org.junit.Test;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * Integration tests to check the correctness of KNN Warmup API
 */

public class RestKNNWarmupHandlerIT extends KNNRestTestCase {

    private static final String RESPONSE_SHARDS = "_shards";
    private static final String RESPONSE_INDICES = "indices";
    private static final String KNN_SHARDS_TOTAL = "total";
    private static final String KNN_SHARDS_SKIPPED = "skipped";
    private static final String KNN_SKIP_REASONS = "skip_reasons";
    private static final String KNN_SKIPPED_FIELDS = "skipped_fields";

    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 2;

    @Test(expected = ResponseException.class)
    public void testNonExistentIndex() throws IOException {
        knnWarmup(Collections.singletonList("non-existent"));
    }

    @Test(expected = ResponseException.class)
    public void testNonKnnIndex() throws IOException {
        createIndex("not-knn-index", Settings.EMPTY);

        knnWarmup(Collections.singletonList("not-knn-index"));
    }

    public void testEmptyIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndexName, getKNNDefaultIndexSettings(), createKnnIndexMapping(testFieldName, dimensions));

        Map<String, Object> responseMap = executeWarmupAndParseResponse(Collections.singletonList(testIndexName));
        Map<String, Object> shards = getShards(responseMap);

        assertEquals(graphCountBefore, getTotalGraphsInCache());
        // The skip decision is mapping based: the non-Lucene k-NN field in the mapping keeps the
        // warmup "executed" even though an empty index has no segments to load yet.
        assertEquals(0, shards.get(KNN_SHARDS_SKIPPED));
        assertNull("No skip means no per-index skip reasons", getIndexStat(responseMap, testIndexName).get(KNN_SKIP_REASONS));
    }

    public void testSingleIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndexName, buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        Map<String, Object> responseMap = executeWarmupAndParseResponse(Collections.singletonList(testIndexName));

        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
        assertEquals(0, getShards(responseMap).get(KNN_SHARDS_SKIPPED));
        Map<String, Object> indexShards = getIndexShards(responseMap, testIndexName);
        assertEquals(1, indexShards.get(KNN_SHARDS_TOTAL));
        assertEquals(0, indexShards.get(KNN_SHARDS_SKIPPED));
    }

    public void testMultipleIndices() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndexName + "1", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "1", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        createKnnIndex(testIndexName + "2", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "2", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        Map<String, Object> responseMap = executeWarmupAndParseResponse(Arrays.asList(testIndexName + "1", testIndexName + "2"));

        assertEquals(graphCountBefore + 2, getTotalGraphsInCache());
        assertEquals(0, getShards(responseMap).get(KNN_SHARDS_SKIPPED));
        assertEquals(1, getIndexShards(responseMap, testIndexName + "1").get(KNN_SHARDS_TOTAL));
        assertEquals(1, getIndexShards(responseMap, testIndexName + "2").get(KNN_SHARDS_TOTAL));
    }

    public void testLuceneEngineIndex_skipsWarmup() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndexName, buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions, METHOD_HNSW, LUCENE_NAME));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        Map<String, Object> responseMap = executeWarmupAndParseResponse(Collections.singletonList(testIndexName));
        Map<String, Object> shards = getShards(responseMap);

        // Lucene engine fields are mmap based; warmup is a no-op and is reported as skipped
        assertEquals(graphCountBefore, getTotalGraphsInCache());
        assertEquals(shards.get(KNN_SHARDS_TOTAL), shards.get(KNN_SHARDS_SKIPPED));

        Map<String, Object> indexShards = getIndexShards(responseMap, testIndexName);
        assertEquals(shards.get(KNN_SHARDS_TOTAL), indexShards.get(KNN_SHARDS_SKIPPED));
        assertEquals(List.of("no_off_heap_warmup"), getIndexStat(responseMap, testIndexName).get(KNN_SKIP_REASONS));
        // All k-NN fields of the index are reported as not loaded
        assertSkippedFields(getIndexStat(responseMap, testIndexName), testFieldName);
    }

    public void testMixedEngineIndex_reportsSkippedFields() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();
        final String luceneFieldName = "lucene-test-field";
        final String faissFieldName = "faiss-test-field";

        createKnnIndex(testIndexName, buildKNNIndexSettings(0), createTwoFieldEngineMapping(luceneFieldName, faissFieldName));
        addKnnDoc(testIndexName, "1", faissFieldName, new Float[] { 6.0f, 6.0f });

        Map<String, Object> responseMap = executeWarmupAndParseResponse(Collections.singletonList(testIndexName));
        Map<String, Object> shards = getShards(responseMap);

        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
        assertEquals(0, shards.get(KNN_SHARDS_SKIPPED));

        Map<String, Object> indexShards = getIndexShards(responseMap, testIndexName);
        assertEquals(0, indexShards.get(KNN_SHARDS_SKIPPED));
        assertNull(getIndexStat(responseMap, testIndexName).get(KNN_SKIP_REASONS));
        assertSkippedFields(getIndexStat(responseMap, testIndexName), luceneFieldName);
    }

    public void testWarmIndex_skipsWarmup() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        // Create a k-NN index with index.warm=true to simulate warm-tier index
        Settings warmSettings = Settings.builder().put(buildKNNIndexSettings(0)).put("index.warm", true).build();
        createKnnIndex(testIndexName, warmSettings, createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        // Warmup should succeed (HTTP 200) but skip loading graphs for the warm index
        Map<String, Object> responseMap = executeWarmupAndParseResponse(Collections.singletonList(testIndexName));
        Map<String, Object> shards = getShards(responseMap);

        // Graph count should not increase because warmup was skipped for warm-tier index
        assertEquals(graphCountBefore, getTotalGraphsInCache());
        assertEquals(shards.get(KNN_SHARDS_TOTAL), shards.get(KNN_SHARDS_SKIPPED));

        Map<String, Object> indexShards = getIndexShards(responseMap, testIndexName);
        assertEquals(shards.get(KNN_SHARDS_TOTAL), indexShards.get(KNN_SHARDS_SKIPPED));
        assertEquals(List.of("warm_tier_index"), getIndexStat(responseMap, testIndexName).get(KNN_SKIP_REASONS));
        // A warm-tier skip is unrelated to fields: no skipped_fields are reported
        assertNull(getIndexStat(responseMap, testIndexName).get(KNN_SKIPPED_FIELDS));
    }

    public void testMixedWarmAndHotIndices_onlyWarmsHotIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        // Create a regular (hot) k-NN index
        createKnnIndex(testIndexName + "-hot", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "-hot", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        // Create a warm-tier k-NN index
        Settings warmSettings = Settings.builder().put(buildKNNIndexSettings(0)).put("index.warm", true).build();
        createKnnIndex(testIndexName + "-warm", warmSettings, createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "-warm", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        // Warmup both indices — only the hot index should have its graphs loaded
        Map<String, Object> responseMap = executeWarmupAndParseResponse(Arrays.asList(testIndexName + "-hot", testIndexName + "-warm"));
        Map<String, Object> shards = getShards(responseMap);

        // Only 1 graph from the hot index should be added; warm index should be skipped
        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
        assertEquals(1, shards.get(KNN_SHARDS_SKIPPED));

        assertEquals(0, getIndexShards(responseMap, testIndexName + "-hot").get(KNN_SHARDS_SKIPPED));
        assertEquals(1, getIndexShards(responseMap, testIndexName + "-warm").get(KNN_SHARDS_SKIPPED));
        assertEquals(List.of("warm_tier_index"), getIndexStat(responseMap, testIndexName + "-warm").get(KNN_SKIP_REASONS));
    }

    private String createTwoFieldEngineMapping(final String luceneFieldName, final String faissFieldName) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(luceneFieldName)
            .field("type", "knn_vector")
            .field("dimension", String.valueOf(dimensions))
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("engine", LUCENE_NAME)
            .endObject()
            .endObject()
            .startObject(faissFieldName)
            .field("type", "knn_vector")
            .field("dimension", String.valueOf(dimensions))
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("engine", "faiss")
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return builder.toString();
    }

    private Map<String, Object> executeWarmupAndParseResponse(List<String> indices) throws IOException, ParseException {
        Response response = knnWarmup(indices);
        assertEquals(200, response.getStatusLine().getStatusCode());
        String responseBody = EntityUtils.toString(response.getEntity());
        return createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getShards(Map<String, Object> responseMap) {
        return (Map<String, Object>) responseMap.get(RESPONSE_SHARDS);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getIndexStat(Map<String, Object> responseMap, String indexName) {
        Map<String, Object> indices = (Map<String, Object>) responseMap.get(RESPONSE_INDICES);
        assertNotNull("Expected per-index details for [" + indexName + "]", indices);
        Map<String, Object> indexStat = (Map<String, Object>) indices.get(indexName);
        assertNotNull("Expected per-index details for [" + indexName + "]", indexStat);
        return indexStat;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getIndexShards(Map<String, Object> responseMap, String indexName) {
        Map<String, Object> indexShards = (Map<String, Object>) getIndexStat(responseMap, indexName).get(RESPONSE_SHARDS);
        assertNotNull("Expected per-index _shards for [" + indexName + "]", indexShards);
        return indexShards;
    }

    @SuppressWarnings("unchecked")
    private void assertSkippedFields(Map<String, Object> indexStat, String... expectedFields) {
        List<String> skippedFields = (List<String>) indexStat.get(KNN_SKIPPED_FIELDS);
        assertNotNull(skippedFields);
        assertEquals(expectedFields.length, skippedFields.size());
        for (String expectedField : expectedFields) {
            assertTrue("Expected skipped field [" + expectedField + "] but got " + skippedFields, skippedFields.contains(expectedField));
        }
    }
}
