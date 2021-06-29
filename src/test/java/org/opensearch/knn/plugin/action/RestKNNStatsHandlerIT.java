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
/*
 *   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package org.opensearch.knn.plugin.action;

import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.KNNQueryBuilder;
import org.opensearch.knn.plugin.stats.KNNStats;
import org.opensearch.knn.plugin.stats.StatNames;
import org.apache.http.util.EntityUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.rules.DisableOnDebug;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.rest.RestStatus;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.plugin.stats.KNNStatsConfig.KNN_STATS;

/**
 * Integration tests to check the correctness of RestKNNStatsHandler
 */
public class RestKNNStatsHandlerIT extends KNNRestTestCase {

    private static final Logger logger = LogManager.getLogger(RestKNNStatsHandlerIT.class);
    private boolean isDebuggingTest = new DisableOnDebug(null).isDebugging();
    private boolean isDebuggingRemoteCluster = System.getProperty("cluster.debug", "false").equals("true");

    private KNNStats knnStats;

    @Before
    public void setup() {
        knnStats = new KNNStats(KNN_STATS);
    }

    /**
     * Test checks that handler correctly returns all metrics
     *
     * @throws IOException throws IOException
     */
    public void testCorrectStatsReturned() throws IOException {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
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
    public void testStatsValueCheck() throws IOException {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats0 = parseNodeStatsResponse(responseBody).get(0);
        Integer hitCount0 = (Integer) nodeStats0.get(StatNames.HIT_COUNT.getName());
        Integer missCount0 = (Integer) nodeStats0.get(StatNames.MISS_COUNT.getName());

        // Setup index
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));

        // Index test document
        Float[] vector = {6.0f, 6.0f};
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // First search: Ensure that misses=1
        float[] qvector = {6.0f, 6.0f};
        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);

        response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats1 = parseNodeStatsResponse(responseBody).get(0);
        Integer hitCount1 = (Integer) nodeStats1.get(StatNames.HIT_COUNT.getName());
        Integer missCount1 = (Integer) nodeStats1.get(StatNames.MISS_COUNT.getName());

        assertEquals(hitCount0, hitCount1);
        assertEquals((Integer) (missCount0 + 1), missCount1);

        // Second search: Ensure that hits=1
        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);

        response = getKnnStats(Collections.emptyList(), Collections.emptyList());
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
    public void testValidMetricsStats() throws IOException {
        // Create request that only grabs two of the possible metrics
        String metric1 = StatNames.HIT_COUNT.getName();
        String metric2 = StatNames.MISS_COUNT.getName();

        Response response = getKnnStats(Collections.emptyList(), Arrays.asList(metric1, metric2));
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
        expectThrows(ResponseException.class, () -> getKnnStats(Collections.emptyList(),
            Collections.singletonList("invalid_metric")));
    }

    //TODO: Fix broken test case
    // This test case intended to check whether the "graph_query_error" stat gets incremented when a query fails.
    // It sets the circuit breaker limit to 1 kb and then indexes documents into the index and force merges so that
    // the sole segment's graph will not fit in the cache. Then, it runs a query and expects an exception. Then it
    // checks that the query errors get incremented. This test is flaky:
    // https://github.com/opensearch-project/k-NN/issues/43. During query, when a segment to be
    // searched is not present in the cache, it will first be loaded. Then it will be searched.
    //
    // The problem is that the cache built from CacheBuilder will not throw an exception if the entry exceeds the
    // size of the cache - tested this via log statements. However, the entry gets marked as expired immediately.
    // So, after loading the entry, sometimes the expired entry will get evicted before the search logic. This causes
    // the search to fail. However, it appears sometimes that the entry doesnt get immediately evicted, causing the
    // search to succeed.
//    public void testGraphQueryErrorsGetIncremented() throws Exception {
//        // Get initial query errors because it may not always be 0
//        String graphQueryErrors = StatNames.GRAPH_QUERY_ERRORS.getName();
//        Response response = getKnnStats(Collections.emptyList(), Collections.singletonList(graphQueryErrors));
//        String responseBody = EntityUtils.toString(response.getEntity());
//        Map<String, Object> nodeStats = parseNodeStatsResponse(responseBody).get(0);
//        int beforeErrors = (int) nodeStats.get(graphQueryErrors);
//
//        // Set the circuit breaker very low so that loading an index will definitely fail
//        updateClusterSettings("knn.memory.circuit_breaker.limit", "1kb");
//
//        Settings settings = Settings.builder()
//                .put("number_of_shards", 1)
//                .put("index.knn", true)
//                .build();
//        createKnnIndex(INDEX_NAME, settings, createKnnIndexMapping(FIELD_NAME, 2));
//
//        // Add enough docs to trip the circuit breaker
//        Float[] vector = {1.3f, 2.2f};
//        int docsInIndex = 25;
//        for (int i = 0; i < docsInIndex; i++) {
//            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, vector);
//        }
//        forceMergeKnnIndex(INDEX_NAME);
//
//        // Execute a query that should fail
//        float[] qvector = {1.9f, 2.4f};
//        expectThrows(ResponseException.class, () ->
//                searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 10), 10));
//
//        // Check that the graphQuery errors gets incremented
//        response = getKnnStats(Collections.emptyList(), Collections.singletonList(graphQueryErrors));
//        responseBody = EntityUtils.toString(response.getEntity());
//        nodeStats = parseNodeStatsResponse(responseBody).get(0);
//        assertTrue((int) nodeStats.get(graphQueryErrors) > beforeErrors);
//    }

    /**
     * Test checks that handler correctly returns stats for a single node
     *
     * @throws IOException throws IOException
     */
    public void testValidNodeIdStats() throws IOException {
        Response response = getKnnStats(Collections.singletonList("_local"), Collections.emptyList());
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
        Response response = getKnnStats(Collections.singletonList("invalid_node"), Collections.emptyList());
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
        Response response = getKnnStats(Collections.emptyList(), Arrays.asList(
            StatNames.SCRIPT_COMPILATIONS.getName(),
            StatNames.SCRIPT_QUERY_REQUESTS.getName(),
            StatNames.SCRIPT_QUERY_ERRORS.getName())
        );
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        int initialScriptCompilations = (int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName()));
        int initialScriptQueryRequests = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName()));
        int initialScriptQueryErrors = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName()));

        // Create an index with a single vector
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] vector = {6.0f, 6.0f};
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // Check l2 query and script compilation stats
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        float[] queryVector = {1.0f, 1.0f};
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.L2.getValue());
        Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        response = getKnnStats(Collections.emptyList(), Arrays.asList(
            StatNames.SCRIPT_COMPILATIONS.getName(),
            StatNames.SCRIPT_QUERY_REQUESTS.getName())
        );
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals((int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName())), initialScriptCompilations + 1);
        assertEquals(initialScriptQueryRequests + 1,
            (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName())));

        // Check query error stats
        params = new HashMap<>();
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", "invalid_space");
        request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        Request finalRequest = request;
        expectThrows(ResponseException.class, () -> client().performRequest(finalRequest));

        response = getKnnStats(Collections.emptyList(), Collections.singletonList(
            StatNames.SCRIPT_QUERY_ERRORS.getName())
        );
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals(initialScriptQueryErrors + 1,
            (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName())));
    }

    /**
     * Test checks that script stats are properly updated for multiple shards
     */
    public void testScriptStats_multipleShards() throws Exception {
        clearScriptCache();

        // Get initial stats
        Response response = getKnnStats(Collections.emptyList(), Arrays.asList(
            StatNames.SCRIPT_COMPILATIONS.getName(),
            StatNames.SCRIPT_QUERY_REQUESTS.getName(),
            StatNames.SCRIPT_QUERY_ERRORS.getName())
        );
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        int initialScriptCompilations = (int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName()));
        int initialScriptQueryRequests = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName()));
        int initialScriptQueryErrors = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName()));

        // Create an index with a single vector
        createKnnIndex(INDEX_NAME, Settings.builder()
                .put("number_of_shards", 2)
                .put("number_of_replicas", 0)
                .put("index.knn", true)
                .build(),
            createKnnIndexMapping(FIELD_NAME, 2));

        Float[] vector = {6.0f, 6.0f};
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, vector);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, vector);
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, vector);

        // Check l2 query and script compilation stats
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        float[] queryVector = {1.0f, 1.0f};
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.L2.getValue());
        Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        response = getKnnStats(Collections.emptyList(), Arrays.asList(
            StatNames.SCRIPT_COMPILATIONS.getName(),
            StatNames.SCRIPT_QUERY_REQUESTS.getName())
        );
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals((int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName())), initialScriptCompilations + 1);
        //TODO fix the test case. For some reason request count is treated as 4.
        // https://github.com/opendistro-for-elasticsearch/k-NN/issues/272
        assertEquals(initialScriptQueryRequests + 4,
            (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName())));

        // Check query error stats
        params = new HashMap<>();
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", "invalid_space");
        request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        Request finalRequest = request;
        expectThrows(ResponseException.class, () -> client().performRequest(finalRequest));

        response = getKnnStats(Collections.emptyList(), Collections.singletonList(
            StatNames.SCRIPT_QUERY_ERRORS.getName())
        );
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals(initialScriptQueryErrors + 2,
            (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName())));
    }

    // Useful settings when debugging to prevent timeouts
    @Override
    protected Settings restClientSettings() {
        if (isDebuggingTest || isDebuggingRemoteCluster) {
            return Settings.builder()
                .put(CLIENT_SOCKET_TIMEOUT, TimeValue.timeValueMinutes(10))
                .build();
        } else {
            return super.restClientSettings();
        }
    }
}
