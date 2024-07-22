/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.action;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.rules.DisableOnDebug;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.cluster.health.ClusterHealthStatus;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.stats.KNNStats;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.core.rest.RestStatus;

import java.io.IOException;
import java.util.*;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.NAME;

/**
 * Integration tests to check the correctness of RestKNNStatsHandler
 */
public class RestKNNStatsHandlerIT extends KNNRestTestCase {

    private static final Logger logger = LogManager.getLogger(RestKNNStatsHandlerIT.class);
    private static final String TRAINING_INDEX = "training-index";
    private static final String TRAINING_FIELD = "training-field";
    private static final String TEST_MODEL_ID = "model-id";
    private static final String TEST_INDEX = "test-index";
    private static final String MODEL_DESCRIPTION = "Description for train model test";
    private boolean isDebuggingTest = new DisableOnDebug(null).isDebugging();
    private boolean isDebuggingRemoteCluster = System.getProperty("cluster.debug", "false").equals("true");
    private static final String FIELD_NAME_2 = "test_field_two";
    private static final String FIELD_NAME_3 = "test_field_three";
    private static final String FIELD_LUCENE_NAME = "lucene_test_field";
    private static final int DIMENSION = 4;
    private static int DOC_ID = 0;
    private static final int NUM_DOCS = 10;
    private static final int DELAY_MILLI_SEC = 1000;
    private static final int NUM_OF_ATTEMPTS = 30;

    private KNNStats knnStats;

    @Before
    public void setup() {
        knnStats = new KNNStats();
    }

    /**
     * Test checks that handler correctly returns all metrics
     *
     * @throws IOException throws IOException
     */
    public void testCorrectStatsReturned() throws Exception {
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
    public void testStatsValueCheck() throws Exception {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats0 = parseNodeStatsResponse(responseBody).get(0);
        Integer hitCount0 = (Integer) nodeStats0.get(StatNames.HIT_COUNT.getName());
        Integer missCount0 = (Integer) nodeStats0.get(StatNames.MISS_COUNT.getName());
        Integer knnQueryWithFilterCount0 = (Integer) nodeStats0.get(StatNames.KNN_QUERY_WITH_FILTER_REQUESTS.getName());

        // Setup index
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));

        // Index test document
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // First search: Ensure that misses=1
        float[] qvector = { 6.0f, 6.0f };
        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);

        response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats1 = parseNodeStatsResponse(responseBody).get(0);
        Integer hitCount1 = (Integer) nodeStats1.get(StatNames.HIT_COUNT.getName());
        Integer missCount1 = (Integer) nodeStats1.get(StatNames.MISS_COUNT.getName());
        Integer knnQueryWithFilterCount1 = (Integer) nodeStats1.get(StatNames.KNN_QUERY_WITH_FILTER_REQUESTS.getName());

        assertEquals(hitCount0, hitCount1);
        assertEquals((Integer) (missCount0 + 1), missCount1);
        assertEquals(knnQueryWithFilterCount0, knnQueryWithFilterCount1);

        // Second search: Ensure that hits=1
        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, qvector, 1), 1);

        response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats2 = parseNodeStatsResponse(responseBody).get(0);
        Integer hitCount2 = (Integer) nodeStats2.get(StatNames.HIT_COUNT.getName());
        Integer missCount2 = (Integer) nodeStats2.get(StatNames.MISS_COUNT.getName());
        Integer knnQueryWithFilterCount2 = (Integer) nodeStats2.get(StatNames.KNN_QUERY_WITH_FILTER_REQUESTS.getName());

        assertEquals(missCount1, missCount2);
        assertEquals((Integer) (hitCount1 + 1), hitCount2);
        assertEquals(knnQueryWithFilterCount0, knnQueryWithFilterCount2);

        putMappingRequest(INDEX_NAME, createKnnIndexMapping(FIELD_LUCENE_NAME, 2, METHOD_HNSW, LUCENE_NAME));
        addKnnDoc(INDEX_NAME, "2", FIELD_LUCENE_NAME, vector);

        searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_LUCENE_NAME, qvector, 1, QueryBuilders.termQuery("_id", "1")), 1);

        response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats3 = parseNodeStatsResponse(responseBody).get(0);
        Integer knnQueryWithFilterCount3 = (Integer) nodeStats3.get(StatNames.KNN_QUERY_WITH_FILTER_REQUESTS.getName());

        assertEquals((Integer) (knnQueryWithFilterCount0 + 1), knnQueryWithFilterCount3);
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
        expectThrows(ResponseException.class, () -> getKnnStats(Collections.emptyList(), Collections.singletonList("invalid_metric")));
    }

    /**
     * Test checks that handler correctly returns stats for a single node
     *
     * @throws IOException throws IOException
     */
    public void testValidNodeIdStats() throws Exception {
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
        Response response = getKnnStats(
            Collections.emptyList(),
            Arrays.asList(
                StatNames.SCRIPT_COMPILATIONS.getName(),
                StatNames.SCRIPT_QUERY_REQUESTS.getName(),
                StatNames.SCRIPT_QUERY_ERRORS.getName()
            )
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

        response = getKnnStats(
            Collections.emptyList(),
            Arrays.asList(StatNames.SCRIPT_COMPILATIONS.getName(), StatNames.SCRIPT_QUERY_REQUESTS.getName())
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

        response = getKnnStats(Collections.emptyList(), Collections.singletonList(StatNames.SCRIPT_QUERY_ERRORS.getName()));
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals(initialScriptQueryErrors + 1, (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName())));
    }

    /**
     * Test checks that script stats are properly updated for multiple shards
     */
    public void testScriptStats_multipleShards() throws Exception {
        clearScriptCache();

        // Get initial stats
        Response response = getKnnStats(
            Collections.emptyList(),
            Arrays.asList(
                StatNames.SCRIPT_COMPILATIONS.getName(),
                StatNames.SCRIPT_QUERY_REQUESTS.getName(),
                StatNames.SCRIPT_QUERY_ERRORS.getName()
            )
        );
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        int initialScriptCompilations = (int) (nodeStats.get(0).get(StatNames.SCRIPT_COMPILATIONS.getName()));
        int initialScriptQueryRequests = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_REQUESTS.getName()));
        int initialScriptQueryErrors = (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName()));

        // Create an index with a single vector
        createKnnIndex(
            INDEX_NAME,
            Settings.builder().put("number_of_shards", 2).put("number_of_replicas", 0).put("index.knn", true).build(),
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

        response = getKnnStats(
            Collections.emptyList(),
            Arrays.asList(StatNames.SCRIPT_COMPILATIONS.getName(), StatNames.SCRIPT_QUERY_REQUESTS.getName())
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

        response = getKnnStats(Collections.emptyList(), Collections.singletonList(StatNames.SCRIPT_QUERY_ERRORS.getName()));
        nodeStats = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        assertEquals(initialScriptQueryErrors + 2, (int) (nodeStats.get(0).get(StatNames.SCRIPT_QUERY_ERRORS.getName())));
    }

    public void testModelIndexHealthMetricsStats() throws Exception {
        String modelIndexStatusName = StatNames.MODEL_INDEX_STATUS.getName();
        // index can be created in one of previous tests, and as we do not delete it each test the check below became optional
        if (!systemIndexExists(MODEL_INDEX_NAME)) {

            final Response response = getKnnStats(Collections.emptyList(), Arrays.asList(modelIndexStatusName));
            final String responseBody = EntityUtils.toString(response.getEntity());
            final Map<String, Object> statsMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

            // Check that model health status is null since model index is not created to system yet
            assertNull(statsMap.get(StatNames.MODEL_INDEX_STATUS.getName()));

            // Train a model so that the system index will get created
            createBasicKnnIndex(TRAINING_INDEX, TRAINING_FIELD, DIMENSION);
            bulkIngestRandomVectors(TRAINING_INDEX, TRAINING_FIELD, NUM_DOCS, DIMENSION);
            trainKnnModel(TEST_MODEL_ID, TRAINING_INDEX, TRAINING_FIELD, DIMENSION, MODEL_DESCRIPTION);
            validateModelCreated(TEST_MODEL_ID);
        }

        Response response = getKnnStats(Collections.emptyList(), Arrays.asList(modelIndexStatusName));

        final String responseBody = EntityUtils.toString(response.getEntity());
        final Map<String, Object> statsMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        // Check that model health status is not null
        assertNotNull(statsMap.get(modelIndexStatusName));

        // Check value is indeed part of ClusterHealthStatus
        assertNotNull(ClusterHealthStatus.fromString((String) statsMap.get(modelIndexStatusName)));

    }

    /**
     * Test checks whether model indexing degraded is available or not
     *
     * @throws IOException throws IOException
     */
    public void testModelIndexingDegradedMetricsStats() throws Exception {
        // Create request that only grabs model indexing degraded stats alone
        String statName = StatNames.INDEXING_FROM_MODEL_DEGRADED.getName();

        Response response = getKnnStats(Collections.emptyList(), Arrays.asList(statName));
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> nodeStats = parseNodeStatsResponse(responseBody).get(0);

        assertTrue("does not contain expected key: " + statName, nodeStats.containsKey(statName));
        assertEquals(false, nodeStats.get(statName));
    }

    /**
     * Test checks that handler correctly returns value for field per engine stats
     *
     * @throws IOException throws IOException
     */
    public void testFieldByEngineStats() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2, METHOD_HNSW, NMSLIB_NAME));
        putMappingRequest(INDEX_NAME, createKnnIndexMapping(FIELD_NAME_2, 3, METHOD_HNSW, LUCENE_NAME));
        putMappingRequest(INDEX_NAME, createKnnIndexMapping(FIELD_NAME_3, 3, METHOD_HNSW, FAISS_NAME));

        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());

        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> nodeStats0 = parseNodeStatsResponse(responseBody).get(0);
        boolean faissField = (Boolean) nodeStats0.get(StatNames.FAISS_LOADED.getName());
        boolean luceneField = (Boolean) nodeStats0.get(StatNames.LUCENE_LOADED.getName());
        boolean nmslibField = (Boolean) nodeStats0.get(StatNames.NMSLIB_LOADED.getName());

        assertTrue(faissField);
        assertTrue(luceneField);
        assertTrue(nmslibField);
    }

    public void testFieldsByEngineModelTraining() throws Exception {
        createBasicKnnIndex(TRAINING_INDEX, TRAINING_FIELD, DIMENSION);
        bulkIngestRandomVectors(TRAINING_INDEX, TRAINING_FIELD, NUM_DOCS, DIMENSION);
        trainKnnModel(TEST_MODEL_ID, TRAINING_INDEX, TRAINING_FIELD, DIMENSION, MODEL_DESCRIPTION);

        validateModelCreated(TEST_MODEL_ID);

        createKnnIndex(TEST_INDEX, modelIndexMapping(FIELD_NAME, TEST_MODEL_ID));

        addKNNDocs(TEST_INDEX, FIELD_NAME, DIMENSION, DOC_ID, NUM_DOCS);

        final Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        final String responseBody = EntityUtils.toString(response.getEntity());

        final Map<String, Object> nodeStats0 = parseNodeStatsResponse(responseBody).get(0);

        boolean faissField = (Boolean) nodeStats0.get(StatNames.FAISS_LOADED.getName());

        assertTrue(faissField);
    }

    public void testRadialSearchStats_thenSucceed() throws Exception {
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2, METHOD_HNSW, LUCENE_NAME));
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // First search: radial search by min score
        XContentBuilder queryBuilderMinScore = XContentFactory.jsonBuilder().startObject().startObject("query");
        queryBuilderMinScore.startObject("knn");
        queryBuilderMinScore.startObject(FIELD_NAME);
        queryBuilderMinScore.field("vector", vector);
        queryBuilderMinScore.field(MIN_SCORE, 0.95f);
        queryBuilderMinScore.endObject();
        queryBuilderMinScore.endObject();
        queryBuilderMinScore.endObject().endObject();

        Integer minScoreStatBeforeMinScoreSearch = getStatCount(StatNames.MIN_SCORE_QUERY_REQUESTS.getName());
        searchKNNIndex(INDEX_NAME, queryBuilderMinScore, 1);
        Integer minScoreStatAfterMinScoreSearch = getStatCount(StatNames.MIN_SCORE_QUERY_REQUESTS.getName());

        assertEquals(1, minScoreStatAfterMinScoreSearch - minScoreStatBeforeMinScoreSearch);

        // Second search: radial search by min score with filter
        XContentBuilder queryBuilderMinScoreWithFilter = XContentFactory.jsonBuilder().startObject().startObject("query");
        queryBuilderMinScoreWithFilter.startObject("knn");
        queryBuilderMinScoreWithFilter.startObject(FIELD_NAME);
        queryBuilderMinScoreWithFilter.field("vector", vector);
        queryBuilderMinScoreWithFilter.field(MIN_SCORE, 0.95f);
        queryBuilderMinScoreWithFilter.field("filter", QueryBuilders.termQuery("_id", "1"));
        queryBuilderMinScoreWithFilter.endObject();
        queryBuilderMinScoreWithFilter.endObject();
        queryBuilderMinScoreWithFilter.endObject().endObject();

        Integer minScoreWithFilterStatBeforeMinScoreWithFilterSearch = getStatCount(
            StatNames.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS.getName()
        );
        Integer minScoreStatBeforeMinScoreWithFilterSearch = getStatCount(StatNames.MIN_SCORE_QUERY_REQUESTS.getName());
        searchKNNIndex(INDEX_NAME, queryBuilderMinScoreWithFilter, 1);
        Integer minScoreWithFilterStatAfterMinScoreWithFilterSearch = getStatCount(
            StatNames.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS.getName()
        );
        Integer minScoreStatAfterMinScoreWithFilterSearch = getStatCount(StatNames.MIN_SCORE_QUERY_REQUESTS.getName());

        assertEquals(1, minScoreWithFilterStatAfterMinScoreWithFilterSearch - minScoreWithFilterStatBeforeMinScoreWithFilterSearch);
        assertEquals(1, minScoreStatAfterMinScoreWithFilterSearch - minScoreStatBeforeMinScoreWithFilterSearch);

        // Third search: radial search by max distance
        XContentBuilder queryBuilderMaxDistance = XContentFactory.jsonBuilder().startObject().startObject("query");
        queryBuilderMaxDistance.startObject("knn");
        queryBuilderMaxDistance.startObject(FIELD_NAME);
        queryBuilderMaxDistance.field("vector", vector);
        queryBuilderMaxDistance.field(MAX_DISTANCE, 100f);
        queryBuilderMaxDistance.endObject();
        queryBuilderMaxDistance.endObject();
        queryBuilderMaxDistance.endObject().endObject();

        Integer maxDistanceStatBeforeMaxDistanceSearch = getStatCount(StatNames.MAX_DISTANCE_QUERY_REQUESTS.getName());
        searchKNNIndex(INDEX_NAME, queryBuilderMaxDistance, 0);
        Integer maxDistanceStatAfterMaxDistanceSearch = getStatCount(StatNames.MAX_DISTANCE_QUERY_REQUESTS.getName());

        assertEquals(1, maxDistanceStatAfterMaxDistanceSearch - maxDistanceStatBeforeMaxDistanceSearch);

        // Fourth search: radial search by max distance with filter
        XContentBuilder queryBuilderMaxDistanceWithFilter = XContentFactory.jsonBuilder().startObject().startObject("query");
        queryBuilderMaxDistanceWithFilter.startObject("knn");
        queryBuilderMaxDistanceWithFilter.startObject(FIELD_NAME);
        queryBuilderMaxDistanceWithFilter.field("vector", vector);
        queryBuilderMaxDistanceWithFilter.field(MAX_DISTANCE, 100f);
        queryBuilderMaxDistanceWithFilter.field("filter", QueryBuilders.termQuery("_id", "1"));
        queryBuilderMaxDistanceWithFilter.endObject();
        queryBuilderMaxDistanceWithFilter.endObject();
        queryBuilderMaxDistanceWithFilter.endObject().endObject();

        Integer maxDistanceWithFilterStatBeforeMaxDistanceWithFilterSearch = getStatCount(
            StatNames.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS.getName()
        );
        Integer maxDistanceStatBeforeMaxDistanceWithFilterSearch = getStatCount(StatNames.MAX_DISTANCE_QUERY_REQUESTS.getName());
        searchKNNIndex(INDEX_NAME, queryBuilderMaxDistanceWithFilter, 0);
        Integer maxDistanceWithFilterStatAfterMaxDistanceWithFilterSearch = getStatCount(
            StatNames.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS.getName()
        );
        Integer maxDistanceStatAfterMaxDistanceWithFilterSearch = getStatCount(StatNames.MAX_DISTANCE_QUERY_REQUESTS.getName());

        assertEquals(
            1,
            maxDistanceWithFilterStatAfterMaxDistanceWithFilterSearch - maxDistanceWithFilterStatBeforeMaxDistanceWithFilterSearch
        );
        assertEquals(1, maxDistanceStatAfterMaxDistanceWithFilterSearch - maxDistanceStatBeforeMaxDistanceWithFilterSearch);
    }

    public void trainKnnModel(String modelId, String trainingIndexName, String trainingFieldName, int dimension, String description)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        Response trainResponse = trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, description);
        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
    }

    public void validateModelCreated(String modelId) throws Exception {
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        assertEquals(modelId, responseMap.get(MODEL_ID));
        assertTrainingSucceeds(modelId, NUM_OF_ATTEMPTS, DELAY_MILLI_SEC);
    }

    // mapping to create index from model
    public String modelIndexMapping(String fieldName, String modelId) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(fieldName)
            .field(VECTOR_TYPE, KNN_VECTOR)
            .field(MODEL_ID, modelId)
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    @Override
    protected boolean preserveClusterUponCompletion() {
        return false;
    }

    // Useful settings when debugging to prevent timeouts
    @Override
    protected Settings restClientSettings() {
        if (isDebuggingTest || isDebuggingRemoteCluster) {
            return Settings.builder().put(CLIENT_SOCKET_TIMEOUT, TimeValue.timeValueMinutes(10)).build();
        } else {
            return super.restClientSettings();
        }
    }

    @SneakyThrows
    private Integer getStatCount(String statName) {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());
        return (Integer) parseNodeStatsResponse(responseBody).get(0).get(statName);
    }
}
