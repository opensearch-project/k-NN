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

        Map<String, Object> shards = executeWarmupAndParseShards(Collections.singletonList(testIndexName));

        assertEquals(graphCountBefore, getTotalGraphsInCache());
        // The skip decision is mapping based: the non-Lucene k-NN field in the mapping keeps the
        // warmup "executed" even though an empty index has no segments to load yet.
        assertEquals(0, shards.get(KNN_SHARDS_SKIPPED));
    }

    public void testSingleIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndexName, buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        Map<String, Object> shards = executeWarmupAndParseShards(Collections.singletonList(testIndexName));

        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
        assertEquals(0, shards.get(KNN_SHARDS_SKIPPED));
    }

    public void testMultipleIndices() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndexName + "1", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "1", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        createKnnIndex(testIndexName + "2", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "2", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        Map<String, Object> shards = executeWarmupAndParseShards(Arrays.asList(testIndexName + "1", testIndexName + "2"));

        assertEquals(graphCountBefore + 2, getTotalGraphsInCache());
        assertEquals(0, shards.get(KNN_SHARDS_SKIPPED));
    }

    public void testLuceneEngineIndex_skipsWarmup() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndexName, buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions, METHOD_HNSW, LUCENE_NAME));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        Map<String, Object> shards = executeWarmupAndParseShards(Collections.singletonList(testIndexName));

        // Lucene engine fields are mmap based; warmup is a no-op and is reported as skipped
        assertEquals(graphCountBefore, getTotalGraphsInCache());
        assertEquals(shards.get(KNN_SHARDS_TOTAL), shards.get(KNN_SHARDS_SKIPPED));
        assertSkipReasons(shards, "lucene_engine");
    }

    public void testWarmIndex_skipsWarmup() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        // Create a k-NN index with index.warm=true to simulate warm-tier index
        Settings warmSettings = Settings.builder().put(buildKNNIndexSettings(0)).put("index.warm", true).build();
        createKnnIndex(testIndexName, warmSettings, createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName, "1", testFieldName, new Float[] { 6.0f, 6.0f });

        // Warmup should succeed (HTTP 200) but skip loading graphs for the warm index
        Map<String, Object> shards = executeWarmupAndParseShards(Collections.singletonList(testIndexName));

        // Graph count should not increase because warmup was skipped for warm-tier index
        assertEquals(graphCountBefore, getTotalGraphsInCache());
        assertEquals(shards.get(KNN_SHARDS_TOTAL), shards.get(KNN_SHARDS_SKIPPED));
        assertSkipReasons(shards, "warm_tier_index");
    }

    public void testMixedWarmAndHotIndices_onlyWarmsHotIndex() throws Exception {
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndexName + "-hot", buildKNNIndexSettings(0), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "-hot", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        Settings warmSettings = Settings.builder().put(buildKNNIndexSettings(0)).put("index.warm", true).build();
        createKnnIndex(testIndexName + "-warm", warmSettings, createKnnIndexMapping(testFieldName, dimensions));
        addKnnDoc(testIndexName + "-warm", "1", testFieldName, new Float[] { 6.0f, 6.0f });

        Map<String, Object> shards = executeWarmupAndParseShards(Arrays.asList(testIndexName + "-hot", testIndexName + "-warm"));

        // Only the hot index graph is loaded; the warm index graph is skipped
        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());
        assertEquals(1, shards.get(KNN_SHARDS_SKIPPED));
        assertSkipReasons(shards, "warm_tier_index");
    }

    private static final String KNN_SHARDS_TOTAL = "total";
    private static final String KNN_SHARDS_SKIPPED = "skipped";
    private static final String KNN_SHARDS_SKIP_REASONS = "skip_reasons";

    @SuppressWarnings("unchecked")
    private Map<String, Object> executeWarmupAndParseShards(List<String> indices) throws IOException, ParseException {
        Response response = knnWarmup(indices);
        assertEquals(200, response.getStatusLine().getStatusCode());
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        return (Map<String, Object>) responseMap.get("_shards");
    }

    @SuppressWarnings("unchecked")
    private void assertSkipReasons(Map<String, Object> shards, String... expectedReasons) {
        List<String> skipReasons = (List<String>) shards.get(KNN_SHARDS_SKIP_REASONS);
        assertNotNull(skipReasons);
        assertEquals(expectedReasons.length, skipReasons.size());
        for (String expectedReason : expectedReasons) {
            assertTrue("Expected skip reason [" + expectedReason + "] but got " + skipReasons, skipReasons.contains(expectedReason));
        }
    }
}
