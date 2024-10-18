/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.knn.KNNRestTestCase;
import org.apache.http.util.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.util.Collections;
import java.util.Map;

import static org.opensearch.knn.index.KNNCircuitBreaker.CB_TIME_INTERVAL;

/**
 * Integration tests to test Circuit Breaker functionality
 */
public class KNNCircuitBreakerIT extends KNNRestTestCase {
    /**
     * To trip the circuit breaker, we will create two indices and index documents. Each index will be small enough so
     * that individually they fit into the cache, but together they do not. To prevent Lucene conditions where
     * multiple segments may or may not be created, we will force merge each index into a single segment before
     * searching.
     */
    private void tripCb() throws Exception {
        // Make sure that Cb is intially not tripped
        assertFalse(isCbTripped());

        // Set circuit breaker limit to 1 KB
        updateClusterSettings("knn.memory.circuit_breaker.limit", "1kb");

        // Create index with 1 primary and numNodes-1 replicas so that the data will be on every node in the cluster
        int numNodes = Integer.parseInt(System.getProperty("cluster.number_of_nodes", "1"));
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", numNodes - 1)
            .put("index.knn", true)
            .build();

        String indexName1 = INDEX_NAME + "1";
        String indexName2 = INDEX_NAME + "2";

        createKnnIndex(indexName1, settings, createKnnIndexMapping(FIELD_NAME, 2));
        createKnnIndex(indexName2, settings, createKnnIndexMapping(FIELD_NAME, 2));

        Float[] vector = { 1.3f, 2.2f };
        int docsInIndex = 7; // through testing, 7 is minimum number of docs to trip circuit breaker at 1kb

        for (int i = 0; i < docsInIndex; i++) {
            addKnnDoc(indexName1, Integer.toString(i), FIELD_NAME, vector);
            addKnnDoc(indexName2, Integer.toString(i), FIELD_NAME, vector);
        }

        forceMergeKnnIndex(indexName1);
        forceMergeKnnIndex(indexName2);

        // Execute search on both indices - will cause eviction
        float[] qvector = { 1.9f, 2.4f };
        int k = 10;

        // Ensure that each shard is searched over so that each Lucene segment gets loaded into memory
        for (int i = 0; i < 15; i++) {
            searchKNNIndex(indexName1, new KNNQueryBuilder(FIELD_NAME, qvector, k), k);
            searchKNNIndex(indexName2, new KNNQueryBuilder(FIELD_NAME, qvector, k), k);
        }

        // Give cluster 5 seconds to update settings and then assert that Cb get triggered
        Thread.sleep(5 * 1000); // seconds
        assertTrue(isCbTripped());
    }

    public boolean isCbTripped() throws Exception {
        Response response = getKnnStats(Collections.emptyList(), Collections.singletonList("circuit_breaker_triggered"));
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> clusterStats = parseClusterStatsResponse(responseBody);
        return Boolean.parseBoolean(clusterStats.get("circuit_breaker_triggered").toString());
    }

    public void testCbTripped() throws Exception {
        tripCb();
    }

    public void testCbUntrips() throws Exception {
        updateClusterSettings("knn.circuit_breaker.triggered", "true");
        assertTrue(isCbTripped());

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
