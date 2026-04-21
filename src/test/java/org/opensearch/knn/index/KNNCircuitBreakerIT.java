/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.hc.client5.http.auth.AuthScope;
import org.apache.hc.client5.http.auth.UsernamePasswordCredentials;
import org.apache.hc.client5.http.impl.auth.BasicCredentialsProvider;
import org.apache.hc.client5.http.impl.nio.PoolingAsyncClientConnectionManager;
import org.apache.hc.client5.http.impl.nio.PoolingAsyncClientConnectionManagerBuilder;
import org.apache.hc.client5.http.ssl.ClientTlsStrategyBuilder;
import org.apache.hc.client5.http.ssl.NoopHostnameVerifier;
import org.apache.hc.core5.http.HttpHost;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.nio.ssl.TlsStrategy;
import org.apache.hc.core5.ssl.SSLContextBuilder;
import org.junit.Assert;
import org.junit.Assume;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.RestClient;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.index.KNNCircuitBreaker.CB_TIME_INTERVAL;

/**
 * Integration tests to test Circuit Breaker functionality
 */
public class KNNCircuitBreakerIT extends KNNRestTestCase {
    private static final Integer ALWAYS_BUILD_GRAPH = 0;
    private static final String INDEX_1 = INDEX_NAME + "1";
    private static final String INDEX_2 = INDEX_NAME + "2";
    private static final String SECURITY_API_PREFIX = "/_plugins/_security/api";
    private static final String SECURITY_AUTH_INFO_ENDPOINT = "/_plugins/_security/authinfo";
    private static final String LIMITED_USER_PASSWORD = "TestUser123!";

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
        return Double.parseDouble(nodeStatsResponse.getFirst().get(StatNames.GRAPH_MEMORY_USAGE_PERCENTAGE.getName()).toString());
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

    public void testCbTrippedWithLimitedUserWhenSecurityEnabled() throws Exception {
        Assume.assumeTrue("Security must be enabled for this test", isSecurityEnabled());

        final String testMethodSuffix = testName.getMethodName().toLowerCase(Locale.ROOT);
        final String roleName = "knn_cb_role_" + testMethodSuffix;
        final String userName = "knn_cb_user_" + testMethodSuffix;

        createLimitedUser(roleName, userName, INDEX_1, INDEX_2);

        try (RestClient limitedUserClient = createLimitedUserClient(userName)) {
            setupIndices();
            updateClusterSettings("knn.memory.circuit_breaker.limit", "1kb");

            float[] qvector = { 1.9f, 2.4f };
            for (int i = 0; i < 15; i++) {
                for (String indexName : new String[] { INDEX_1, INDEX_2 }) {
                    Request request = new Request("POST", "/" + indexName + "/_search");
                    request.setJsonEntity(
                        XContentFactory.jsonBuilder()
                            .startObject()
                            .startObject("query")
                            .startObject("knn")
                            .startObject(FIELD_NAME)
                            .array("vector", qvector[0], qvector[1])
                            .field("k", 10)
                            .endObject()
                            .endObject()
                            .endObject()
                            .endObject()
                            .toString()
                    );
                    Response response = limitedUserClient.performRequest(request);
                    assertEquals(RestStatus.OK.getStatus(), response.getStatusLine().getStatusCode());
                }
            }

            assertBusy(() -> assertTrue("Circuit breaker should trip even for a non-admin user", isCbTripped()), 30, TimeUnit.SECONDS);
        } finally {
            adminClient().performRequest(new Request("DELETE", SECURITY_API_PREFIX + "/internalusers/" + userName));
            adminClient().performRequest(new Request("DELETE", SECURITY_API_PREFIX + "/rolesmapping/" + roleName));
            adminClient().performRequest(new Request("DELETE", SECURITY_API_PREFIX + "/roles/" + roleName));
        }
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

    private boolean isSecurityEnabled() {
        try {
            Response response = adminClient().performRequest(new Request("GET", SECURITY_AUTH_INFO_ENDPOINT));
            return response.getStatusLine().getStatusCode() == RestStatus.OK.getStatus();
        } catch (Exception e) {
            return false;
        }
    }

    private void createLimitedUser(String roleName, String userName, String... indices) throws IOException {
        Request roleRequest = new Request("PUT", SECURITY_API_PREFIX + "/roles/" + roleName);
        roleRequest.setJsonEntity(
            XContentFactory.jsonBuilder()
                .startObject()
                .array("cluster_permissions", "cluster_composite_ops_ro")
                .startArray("index_permissions")
                .startObject()
                .array("index_patterns", indices)
                .array("allowed_actions", "read")
                .endObject()
                .endArray()
                .endObject()
                .toString()
        );
        assertOK(adminClient().performRequest(roleRequest));

        Request userRequest = new Request("PUT", SECURITY_API_PREFIX + "/internalusers/" + userName);
        userRequest.setJsonEntity(
            XContentFactory.jsonBuilder()
                .startObject()
                .field("password", LIMITED_USER_PASSWORD)
                .array("backend_roles", roleName)
                .endObject()
                .toString()
        );
        assertOK(adminClient().performRequest(userRequest));

        Request mappingRequest = new Request("PUT", SECURITY_API_PREFIX + "/rolesmapping/" + roleName);
        mappingRequest.setJsonEntity(
            XContentFactory.jsonBuilder().startObject().array("users", userName).array("backend_roles", roleName).endObject().toString()
        );
        assertOK(adminClient().performRequest(mappingRequest));
    }

    private RestClient createLimitedUserClient(String userName) {
        BasicCredentialsProvider credentialsProvider = new BasicCredentialsProvider();
        credentialsProvider.setCredentials(
            new AuthScope(null, -1),
            new UsernamePasswordCredentials(userName, LIMITED_USER_PASSWORD.toCharArray())
        );
        try {
            TlsStrategy tlsStrategy = ClientTlsStrategyBuilder.create()
                .setHostnameVerifier(NoopHostnameVerifier.INSTANCE)
                .setSslContext(SSLContextBuilder.create().loadTrustMaterial(null, (chains, authType) -> true).build())
                .build();
            PoolingAsyncClientConnectionManager connectionManager = PoolingAsyncClientConnectionManagerBuilder.create()
                .setTlsStrategy(tlsStrategy)
                .build();
            return RestClient.builder(getClusterHosts().toArray(new HttpHost[0]))
                .setHttpClientConfigCallback(
                    httpClientBuilder -> httpClientBuilder.setDefaultCredentialsProvider(credentialsProvider)
                        .setConnectionManager(connectionManager)
                )
                .build();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
