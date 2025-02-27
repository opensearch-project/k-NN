/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.action;

import org.opensearch.Version;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.stats.KNNStats;
import org.opensearch.knn.plugin.stats.StatNames;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Before;
import org.junit.rules.DisableOnDebug;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.core.rest.RestStatus;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Integration tests to check the correctness of Legacy stats api
 */
public class RestLegacyKNNStatsHandlerIT extends KNNRestTestCase {

    private boolean isDebuggingTest = new DisableOnDebug(null).isDebugging();
    private boolean isDebuggingRemoteCluster = System.getProperty("cluster.debug", "false").equals("true");

    private KNNStats knnStats;

    @Before
    public void setup() {
        knnStats = new KNNStats(null, () -> Version.CURRENT);
    }

    /**
     * Test checks that handler correctly returns all metrics
     *
     * @throws IOException throws IOException
     */
    public void testCorrectStatsReturned() throws Exception {
        Response response = executeKnnStatRequest(Collections.emptyList(), Collections.emptyList(), KNNPlugin.LEGACY_KNN_BASE_URI);
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> clusterStats = parseClusterStatsResponse(responseBody);
        assertEquals(knnStats.getClusterStats().keySet(), clusterStats.keySet());
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(responseBody);
        assertEquals(knnStats.getNodeStats().keySet(), nodeStats.get(0).keySet());
    }

    /**
     * Test checks that handler correctly returns value for select metrics
     *
     * @throws IOException throws IOException
     */
    public void testStatsValueCheck() throws Exception {
        Response response = executeKnnStatRequest(Collections.emptyList(), Collections.emptyList(), KNNPlugin.LEGACY_KNN_BASE_URI);
        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats0 = parseNodeStatsResponse(responseBody).get(0);
        Integer hitCount0 = (Integer) nodeStats0.get(StatNames.HIT_COUNT.getName());
        Integer missCount0 = (Integer) nodeStats0.get(StatNames.MISS_COUNT.getName());

        // Setup index
        createKnnIndex(INDEX_NAME, buildKNNIndexSettings(0), createKnnIndexMapping(FIELD_NAME, 2));

        // Index test document
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // First search: Ensure that misses=1
        float[] qvector = { 6.0f, 6.0f };
        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);

        response = executeKnnStatRequest(Collections.emptyList(), Collections.emptyList(), KNNPlugin.LEGACY_KNN_BASE_URI);
        responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats1 = parseNodeStatsResponse(responseBody).get(0);
        Integer hitCount1 = (Integer) nodeStats1.get(StatNames.HIT_COUNT.getName());
        Integer missCount1 = (Integer) nodeStats1.get(StatNames.MISS_COUNT.getName());

        assertEquals(hitCount0, hitCount1);
        assertEquals((Integer) (missCount0 + 1), missCount1);

        // Second search: Ensure that hits=1
        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);

        response = executeKnnStatRequest(Collections.emptyList(), Collections.emptyList(), KNNPlugin.LEGACY_KNN_BASE_URI);
        responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats2 = parseNodeStatsResponse(responseBody).get(0);
        Integer hitCount2 = (Integer) nodeStats2.get(StatNames.HIT_COUNT.getName());
        Integer missCount2 = (Integer) nodeStats2.get(StatNames.MISS_COUNT.getName());

        assertEquals(missCount1, missCount2);
        assertEquals((Integer) (hitCount1 + 1), hitCount2);
    }

    /**
     * Test checks that handler correctly returns selected metrics
     *
     * @throws IOException throws IOException
     */
    public void testValidMetricsStats() throws Exception {
        // Create request that only grabs two of the possible metrics
        String metric1 = StatNames.HIT_COUNT.getName();
        String metric2 = StatNames.MISS_COUNT.getName();

        Response response = executeKnnStatRequest(Collections.emptyList(), Arrays.asList(metric1, metric2), KNNPlugin.LEGACY_KNN_BASE_URI);
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> nodeStats = parseNodeStatsResponse(responseBody).get(0);

        // Check that metric 1 and 2 are the only metrics in the response
        assertEquals("Incorrect number of metrics returned", 2, nodeStats.size());
        assertTrue("does not contain correct metric: " + metric1, nodeStats.containsKey(metric1));
        assertTrue("does not contain correct metric: " + metric2, nodeStats.containsKey(metric2));
    }

    /**
     * Test checks that handler correctly returns failure on an invalid metric
     */
    public void testInvalidMetricsStats() {
        expectThrows(
            ResponseException.class,
            () -> executeKnnStatRequest(Collections.emptyList(), Collections.singletonList("invalid_metric"), KNNPlugin.LEGACY_KNN_BASE_URI)
        );
    }

    /**
     * Test checks that handler correctly returns stats for a single node
     *
     * @throws IOException throws IOException
     */
    public void testValidNodeIdStats() throws Exception {
        Response response = executeKnnStatRequest(
            Collections.singletonList("_local"),
            Collections.emptyList(),
            KNNPlugin.LEGACY_KNN_BASE_URI
        );
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(responseBody);
        assertEquals(1, nodeStats.size());
    }

    /**
     * Test checks that handler correctly returns failure on an invalid node
     *
     * @throws Exception throws Exception
     */
    public void testInvalidNodeIdStats() throws Exception {
        Response response = executeKnnStatRequest(
            Collections.singletonList("invalid_node"),
            Collections.emptyList(),
            KNNPlugin.LEGACY_KNN_BASE_URI
        );
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(responseBody);
        assertEquals(0, nodeStats.size());
    }

    /**
     * Test checks that script stats are properly updated for single shard
     */
    public void testScriptStats_singleShard() throws Exception {
        clearScriptCache();

        // Get initial stats
        Response response = executeKnnStatRequest(
            Collections.emptyList(),
            Arrays.asList(
                StatNames.SCRIPT_COMPILATIONS.getName(),
                StatNames.SCRIPT_QUERY_REQUESTS.getName(),
                StatNames.SCRIPT_QUERY_ERRORS.getName()
            ),
            KNNPlugin.LEGACY_KNN_BASE_URI
        );
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        int initialScriptCompilations = (int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName()));
        int initialScriptQueryRequests = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName()));
        int initialScriptQueryErrors = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName()));

        // Create an index with a single vector
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // Check l2 query and script compilation stats
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        float[] queryVector = { 1.0f, 1.0f };
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.L2.getValue());
        Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        response = executeKnnStatRequest(
            Collections.emptyList(),
            Arrays.asList(StatNames.SCRIPT_COMPILATIONS.getName(), StatNames.SCRIPT_QUERY_REQUESTS.getName()),
            KNNPlugin.LEGACY_KNN_BASE_URI
        );
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals((int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName())), initialScriptCompilations + 1);
        assertEquals(initialScriptQueryRequests + 1, (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName())));

        // Check query error stats
        params = new HashMap<>();
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", "invalid_space");
        request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        Request finalRequest = request;
        expectThrows(ResponseException.class, () -> client().performRequest(finalRequest));

        response = executeKnnStatRequest(
            Collections.emptyList(),
            Collections.singletonList(StatNames.SCRIPT_QUERY_ERRORS.getName()),
            KNNPlugin.LEGACY_KNN_BASE_URI
        );
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals(initialScriptQueryErrors + 1, (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName())));
    }

    /**
     * Test checks that script stats are properly updated for multiple shards
     */
    public void testScriptStats_multipleShards() throws Exception {
        clearScriptCache();

        // Get initial stats
        Response response = executeKnnStatRequest(
            Collections.emptyList(),
            Arrays.asList(
                StatNames.SCRIPT_COMPILATIONS.getName(),
                StatNames.SCRIPT_QUERY_REQUESTS.getName(),
                StatNames.SCRIPT_QUERY_ERRORS.getName()
            ),
            KNNPlugin.LEGACY_KNN_BASE_URI
        );
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        int initialScriptCompilations = (int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName()));
        int initialScriptQueryRequests = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName()));
        int initialScriptQueryErrors = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName()));

        // Create an index with a single vector
        createKnnIndex(
            INDEX_NAME,
            Settings.builder()
                .put("number_of_shards", 2)
                .put("number_of_replicas", 0)
                .put("index.knn", true)
                .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
                .build(),
            createKnnIndexMapping(FIELD_NAME, 2)
        );

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, vector);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, vector);
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, vector);

        // Check l2 query and script compilation stats
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        float[] queryVector = { 1.0f, 1.0f };
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.L2.getValue());
        Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        response = executeKnnStatRequest(
            Collections.emptyList(),
            Arrays.asList(StatNames.SCRIPT_COMPILATIONS.getName(), StatNames.SCRIPT_QUERY_REQUESTS.getName()),
            KNNPlugin.LEGACY_KNN_BASE_URI
        );
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals((int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName())), initialScriptCompilations + 1);
        // TODO fix the test case. For some reason request count is treated as 4.
        // https://github.com/opendistro-for-elasticsearch/k-NN/issues/272
        assertEquals(initialScriptQueryRequests + 4, (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName())));

        // Check query error stats
        params = new HashMap<>();
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", "invalid_space");
        request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        Request finalRequest = request;
        expectThrows(ResponseException.class, () -> client().performRequest(finalRequest));

        response = executeKnnStatRequest(
            Collections.emptyList(),
            Collections.singletonList(StatNames.SCRIPT_QUERY_ERRORS.getName()),
            KNNPlugin.LEGACY_KNN_BASE_URI
        );
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals(initialScriptQueryErrors + 2, (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName())));
    }

    // Useful settings when debugging to prevent timeouts
    @Override
    protected Settings restClientSettings() {
        final Settings.Builder builder = Settings.builder();
        if (isDebuggingTest || isDebuggingRemoteCluster) {
            builder.put(CLIENT_SOCKET_TIMEOUT, TimeValue.timeValueMinutes(10));
        } else {
            if (System.getProperty("tests.rest.client_path_prefix") != null) {
                builder.put(CLIENT_PATH_PREFIX, System.getProperty("tests.rest.client_path_prefix"));
            }
        }
        builder.put("strictDeprecationMode", false);
        return builder.build();
    }
}
