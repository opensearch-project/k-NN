/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.junit.Assert;
import org.opensearch.knn.KNNRestTestCase;
import org.apache.http.util.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.KNNCircuitBreaker.CB_TIME_INTERVAL;

/**
 * Integration tests to test Circuit Breaker functionality
 */
public class KNNCircuitBreakerIT extends KNNRestTestCase {
    private static final Integer ALWAYS_BUILD_GRAPH = 0;
    private static final String INDEX_1 = INDEX_NAME + "1";
    private static final String INDEX_2 = INDEX_NAME + "2";

    /**
     * Base setup for all circuit breaker tests.
     * Creates two indices with enough documents to consume ~2kb of memory when loaded.
     */
    private void setupIndices() throws Exception {
        generateCbLoad(INDEX_1, INDEX_2);

        // Verify initial state
        assertFalse(isCbTripped());
    }

    /**
     * Tests circuit breaker behavior with only cluster-level limit configured.
     * Expected behavior:
     * 1. Set cluster limit to 1kb
     * 2. Load indices consuming 2kb
     * 3. Circuit breaker should trip
     */
    private void testClusterLevelCircuitBreaker() throws Exception {
        // Set cluster-level limit to 1kb (half of what indices require)
        updateClusterSettings("knn.memory.circuit_breaker.limit", "1kb");

        // Load indices into cache
        search(INDEX_1, INDEX_2);

        // Verify circuit breaker tripped
        Thread.sleep(5 * 1000);
        assertTrue(isCbTripped());
    }

    /**
    * Tests circuit breaker behavior with only node-level limit configured.
    * Expected behavior:
    * 1. Set cluster limit high (10kb) to ensure it doesn't interfere
    * 2. Set node limit to 1kb
    * 3. Load indices consuming 2kb
    * 4. Circuit breaker should trip because node limit (1kb) < memory needed (2kb)
    */
    private void testNodeLevelCircuitBreaker() throws Exception {

        // Set cluster-level limit high to ensure it doesn't interfere
        updateClusterSettings("knn.memory.circuit_breaker.limit", "10kb");

        // Set node-level limit to 1kb (half of what indices require)
        updateClusterSettings("knn.memory.circuit_breaker.limit.integ", "1kb");

        // Load indices into cache
        search(INDEX_1, INDEX_2);

        // Verify circuit breaker tripped with 1kb node limit
        Thread.sleep(5 * 1000);
        assertTrue(isCbTripped());

        // Increase node limit to 4kb - should untrip CB and show 50% usage
        updateClusterSettings("knn.memory.circuit_breaker.limit.integ", "4kb");

        // Load indices again
        search(INDEX_1, INDEX_2);

        // Verify CB untripped and correct memory usage
        Thread.sleep(5 * 1000);
        verifyCbUntrips();

        // The contents of the cache should take about 2kb with the current test.
        // This could change in the future depending on the cache library and other factors
        // Verify value is greater than what the percentage would be if the cluster level circuit breaker was in play with 2kb
        Assert.assertTrue(getGraphMemoryPercentage() > 20.0);
    }

    private void generateCbLoad(String indexName1, String indexName2) throws Exception {
        // Create index with 1 primary and numNodes-1 replicas so that the data will be on every node in the cluster
        int numNodes = Integer.parseInt(System.getProperty("cluster.number_of_nodes", "1"));
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", numNodes - 1)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, ALWAYS_BUILD_GRAPH)
            .build();

        createKnnIndex(indexName1, settings, createKnnIndexMapping(FIELD_NAME, 2));
        createKnnIndex(indexName2, settings, createKnnIndexMapping(FIELD_NAME, 2));

        Float[] vector = { 1.3f, 2.2f };
        int docsInIndex = 10; // through testing, 10 is minimum number of docs to trip circuit breaker at 1kb

        for (int i = 0; i < docsInIndex; i++) {
            addKnnDoc(indexName1, Integer.toString(i), FIELD_NAME, vector);
            addKnnDoc(indexName2, Integer.toString(i), FIELD_NAME, vector);
        }

        forceMergeKnnIndex(indexName1);
        forceMergeKnnIndex(indexName2);
    }

    private void search(String indexName1, String indexName2) throws IOException {
        // Execute search on both indices - will cause eviction
        float[] qvector = { 1.9f, 2.4f };
        int k = 10;

        // Ensure that each shard is searched over so that each Lucene segment gets loaded into memory
        for (int i = 0; i < 15; i++) {
            searchKNNIndex(indexName1, new KNNQueryBuilder(FIELD_NAME, qvector, k), k);
            searchKNNIndex(indexName2, new KNNQueryBuilder(FIELD_NAME, qvector, k), k);
        }
    }

    private double getGraphMemoryPercentage() throws Exception {
        Response response = getKnnStats(
            Collections.emptyList(),
            Collections.singletonList(StatNames.GRAPH_MEMORY_USAGE_PERCENTAGE.getName())
        );
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> nodeStatsResponse = parseNodeStatsResponse(responseBody);
        return Double.parseDouble(nodeStatsResponse.get(0).get(StatNames.GRAPH_MEMORY_USAGE_PERCENTAGE.getName()).toString());
    }

    public boolean isCbTripped() throws Exception {
        Response response = getKnnStats(Collections.emptyList(), Collections.singletonList("circuit_breaker_triggered"));
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> clusterStats = parseClusterStatsResponse(responseBody);
        return Boolean.parseBoolean(clusterStats.get("circuit_breaker_triggered").toString());
    }

    @ExpectRemoteBuildValidation
    public void testCbTripped() throws Exception {
        setupIndices();
        testClusterLevelCircuitBreaker();
        testNodeLevelCircuitBreaker();
    }

    public void verifyCbUntrips() throws Exception {

        if (!isCbTripped()) {
            updateClusterSettings("knn.circuit_breaker.triggered", "true");

        }

        int backOffInterval = 5; // seconds
        for (int i = 0; i < CB_TIME_INTERVAL; i += backOffInterval) {
            if (!isCbTripped()) {
                break;
            }
            Thread.sleep(backOffInterval * 1000);
        }
        assertFalse(isCbTripped());
    }
}
