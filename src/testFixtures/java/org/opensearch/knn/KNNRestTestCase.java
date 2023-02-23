/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import com.google.common.primitives.Floats;
import org.apache.commons.lang.StringUtils;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.script.KNNScoringScriptEngine;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.AfterClass;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.Strings;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.index.query.ExistsQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreQueryBuilder;
import org.opensearch.rest.RestStatus;
import org.opensearch.script.Script;
import org.opensearch.search.aggregations.metrics.ScriptedMetricAggregationBuilder;

import javax.management.MBeanServerInvocationHandler;
import javax.management.MalformedObjectNameException;
import javax.management.ObjectName;
import javax.management.remote.JMXConnector;
import javax.management.remote.JMXConnectorFactory;
import javax.management.remote.JMXServiceURL;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ERROR;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_MAPPING_PATH;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.common.KNNConstants.MODEL_TIMESTAMP;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

import static org.opensearch.knn.TestUtils.NUMBER_OF_REPLICAS;
import static org.opensearch.knn.TestUtils.NUMBER_OF_SHARDS;
import static org.opensearch.knn.TestUtils.INDEX_KNN;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.FIELD;
import static org.opensearch.knn.TestUtils.QUERY_VALUE;
import static org.opensearch.knn.TestUtils.computeGroundTruthValues;

import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.GRAPH_COUNT;
import static org.opensearch.knn.plugin.stats.StatNames.INDICES_IN_CACHE;

/**
 * Base class for integration tests for KNN plugin. Contains several methods for testing KNN ES functionality.
 */
public class KNNRestTestCase extends ODFERestTestCase {
    public static final String INDEX_NAME = "test_index";
    public static final String FIELD_NAME = "test_field";
    private static final String DOCUMENT_FIELD_SOURCE = "_source";
    private static final String DOCUMENT_FIELD_FOUND = "found";

    @AfterClass
    public static void dumpCoverage() throws IOException, MalformedObjectNameException {
        // jacoco.dir is set in esplugin-coverage.gradle, if it doesn't exist we don't
        // want to collect coverage so we can return early
        String jacocoBuildPath = System.getProperty("jacoco.dir");
        if (Strings.isNullOrEmpty(jacocoBuildPath)) {
            return;
        }

        String serverUrl = System.getProperty("jmx.serviceUrl");
        try (JMXConnector connector = JMXConnectorFactory.connect(new JMXServiceURL(serverUrl))) {
            IProxy proxy = MBeanServerInvocationHandler.newProxyInstance(
                connector.getMBeanServerConnection(),
                new ObjectName("org.jacoco:type=Runtime"),
                IProxy.class,
                false
            );

            Path path = Paths.get(jacocoBuildPath + "/integTest.exec");
            Files.write(path, proxy.getExecutionData(false));
        } catch (Exception ex) {
            throw new RuntimeException("Failed to dump coverage: " + ex);
        }
    }

    @Before
    public void cleanUpCache() throws Exception {
        clearCache();
    }

    /**
     * Create KNN Index with default settings
     */
    protected void createKnnIndex(String index, String mapping) throws IOException {
        createIndex(index, getKNNDefaultIndexSettings());
        putMappingRequest(index, mapping);
    }

    /**
     * Create KNN Index
     */
    protected void createKnnIndex(String index, Settings settings, String mapping) throws IOException {
        createIndex(index, settings);
        putMappingRequest(index, mapping);
    }

    protected void createBasicKnnIndex(String index, String fieldName, int dimension) throws IOException {
        String mapping = Strings.toString(
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(fieldName)
                .field("type", "knn_vector")
                .field("dimension", Integer.toString(dimension))
                .endObject()
                .endObject()
                .endObject()
        );

        mapping = mapping.substring(1, mapping.length() - 1);
        createIndex(index, Settings.EMPTY, mapping);
    }

    /**
     * Run KNN Search on Index
     */
    protected Response searchKNNIndex(String index, KNNQueryBuilder knnQueryBuilder, int resultSize) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnQueryBuilder.doXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject().endObject();

        Request request = new Request("POST", "/" + index + "/_search");

        request.addParameter("size", Integer.toString(resultSize));
        request.addParameter("explain", Boolean.toString(true));
        request.addParameter("search_type", "query_then_fetch");
        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }

    /**
     * Run exists search
     */
    protected Response searchExists(String index, ExistsQueryBuilder existsQueryBuilder, int resultSize) throws IOException {

        Request request = new Request("POST", "/" + index + "/_search");

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        builder = XContentFactory.jsonBuilder().startObject();
        builder.field("query", existsQueryBuilder);
        builder.endObject();

        request.addParameter("size", Integer.toString(resultSize));
        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }

    /**
     * Parse the response of KNN search into a List of KNNResults
     */
    protected List<KNNResult> parseSearchResponse(String responseBody, String fieldName) throws IOException {
        @SuppressWarnings("unchecked")
        List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(XContentType.JSON.xContent(), responseBody).map()
            .get("hits")).get("hits");

        @SuppressWarnings("unchecked")
        List<KNNResult> knnSearchResponses = hits.stream().map(hit -> {
            @SuppressWarnings("unchecked")
            Float[] vector = Arrays.stream(
                ((ArrayList<Float>) ((Map<String, Object>) ((Map<String, Object>) hit).get("_source")).get(fieldName)).toArray()
            ).map(Object::toString).map(Float::valueOf).toArray(Float[]::new);
            return new KNNResult((String) ((Map<String, Object>) hit).get("_id"), vector);
        }).collect(Collectors.toList());

        return knnSearchResponses;
    }

    protected List<Float> parseSearchResponseScore(String responseBody, String fieldName) throws IOException {
        @SuppressWarnings("unchecked")
        List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(XContentType.JSON.xContent(), responseBody).map()
            .get("hits")).get("hits");

        @SuppressWarnings("unchecked")
        List<Float> knnSearchResponses = hits.stream()
            .map(hit -> ((Double) ((Map<String, Object>) hit).get("_score")).floatValue())
            .collect(Collectors.toList());

        return knnSearchResponses;
    }

    /**
     * Parse the response of Aggregation to extract the value
     */
    protected Double parseAggregationResponse(String responseBody, String aggregationName) throws IOException {
        @SuppressWarnings("unchecked")
        Map<String, Object> aggregations = ((Map<String, Object>) createParser(XContentType.JSON.xContent(), responseBody).map()
            .get("aggregations"));

        final Map<String, Object> values = (Map<String, Object>) aggregations.get(aggregationName);
        return Double.valueOf(String.valueOf(values.get("value")));
    }

    /**
     * Parse the score from the KNN search response
     */

    /**
     * Delete KNN index
     */
    protected void deleteKNNIndex(String index) throws IOException {
        Request request = new Request("DELETE", "/" + index);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * For a given index, make a mapping request
     */
    protected void putMappingRequest(String index, String mapping) throws IOException {
        // Put KNN mapping
        Request request = new Request("PUT", "/" + index + "/_mapping");

        request.setJsonEntity(mapping);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Utility to create a Knn Index Mapping
     */
    protected String createKnnIndexMapping(String fieldName, Integer dimensions) throws IOException {
        return Strings.toString(
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(fieldName)
                .field("type", "knn_vector")
                .field("dimension", dimensions.toString())
                .endObject()
                .endObject()
                .endObject()
        );
    }

    /**
     * Utility to create a Knn Index Mapping with specific algorithm and engine
     */
    protected String createKnnIndexMapping(String fieldName, Integer dimensions, String algoName, String knnEngine) throws IOException {
        return Strings.toString(
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(fieldName)
                .field("type", "knn_vector")
                .field("dimension", dimensions.toString())
                .startObject("method")
                .field("name", algoName)
                .field("engine", knnEngine)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
        );
    }

    /**
     * Utility to create a Knn Index Mapping with multiple k-NN fields
     */
    protected String createKnnIndexMapping(List<String> fieldNames, List<Integer> dimensions) throws IOException {
        assertNotEquals(0, fieldNames.size());
        assertEquals(fieldNames.size(), dimensions.size());

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().startObject("properties");
        for (int i = 0; i < fieldNames.size(); i++) {
            xContentBuilder.startObject(fieldNames.get(i))
                .field("type", "knn_vector")
                .field("dimension", dimensions.get(i).toString())
                .endObject();
        }
        xContentBuilder.endObject().endObject();

        return Strings.toString(xContentBuilder);
    }

    /**
     * Get index mapping as map
     *
     * @param index name of index to fetch
     * @return index mapping a map
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> getIndexMappingAsMap(String index) throws Exception {
        Request request = new Request("GET", "/" + index + "/_mapping");

        Response response = client().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();

        return (Map<String, Object>) ((Map<String, Object>) responseMap.get(index)).get("mappings");
    }

    public int getDocCount(String indexName) throws Exception {
        Request request = new Request("GET", "/" + indexName + "/_count");

        Response response = client().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();
        return (Integer) responseMap.get("count");
    }

    /**
     * Force merge KNN index segments
     */
    protected void forceMergeKnnIndex(String index) throws Exception {
        Request request = new Request("POST", "/" + index + "/_refresh");

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        request = new Request("POST", "/" + index + "/_forcemerge");

        request.addParameter("max_num_segments", "1");
        request.addParameter("flush", "true");
        response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        TimeUnit.SECONDS.sleep(5); // To make sure force merge is completed
    }

    /**
     * Add a single KNN Doc to an index
     */
    protected void addKnnDoc(String index, String docId, String fieldName, Object[] vector) throws IOException {
        Request request = new Request("POST", "/" + index + "/_doc/" + docId + "?refresh=true");

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field(fieldName, vector).endObject();
        request.setJsonEntity(Strings.toString(builder));
        client().performRequest(request);

        request = new Request("POST", "/" + index + "/_refresh");
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Add a single KNN Doc to an index with multiple fields
     */
    protected void addKnnDoc(String index, String docId, List<String> fieldNames, List<Object[]> vectors) throws IOException {
        Request request = new Request("POST", "/" + index + "/_doc/" + docId + "?refresh=true");

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        for (int i = 0; i < fieldNames.size(); i++) {
            builder.field(fieldNames.get(i), vectors.get(i));
        }
        builder.endObject();

        request.setJsonEntity(Strings.toString(builder));
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Add a single numeric field Doc to an index
     */
    protected void addDocWithNumericField(String index, String docId, String fieldName, long value) throws IOException {
        Request request = new Request("POST", "/" + index + "/_doc/" + docId + "?refresh=true");

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field(fieldName, value).endObject();

        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Add a single numeric field Doc to an index
     */
    protected void addDocWithBinaryField(String index, String docId, String fieldName, String base64String) throws IOException {
        Request request = new Request("POST", "/" + index + "/_doc/" + docId + "?refresh=true");

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field(fieldName, base64String).endObject();

        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Update a KNN Doc with a new vector for the given fieldName
     */
    protected void updateKnnDoc(String index, String docId, String fieldName, Object[] vector) throws IOException {
        Request request = new Request("POST", "/" + index + "/_doc/" + docId + "?refresh=true");

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field(fieldName, vector).endObject();

        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Delete Knn Doc
     */
    protected void deleteKnnDoc(String index, String docId) throws IOException {
        // Put KNN mapping
        Request request = new Request("DELETE", "/" + index + "/_doc/" + docId + "?refresh");

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Retrieve document by index and document id
     */
    protected Map<String, Object> getKnnDoc(final String index, final String docId) throws Exception {
        final Request request = new Request("GET", "/" + index + "/_doc/" + docId);
        final Response response = client().performRequest(request);

        final Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), EntityUtils.toString(response.getEntity()))
            .map();

        assertNotNull(responseMap);
        assertTrue((Boolean) responseMap.get(DOCUMENT_FIELD_FOUND));
        assertNotNull(responseMap.get(DOCUMENT_FIELD_SOURCE));

        final Map<String, Object> docMap = (Map<String, Object>) responseMap.get(DOCUMENT_FIELD_SOURCE);

        return docMap;
    }

    /**
     * Utility to update  settings
     */
    protected void updateClusterSettings(String settingKey, Object value) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("persistent")
            .field(settingKey, value)
            .endObject()
            .endObject();
        Request request = new Request("PUT", "_cluster/settings");
        request.setJsonEntity(Strings.toString(builder));
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Return default index settings for index creation
     */
    protected Settings getKNNDefaultIndexSettings() {
        return Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", true).build();
    }

    /**
     * Get Stats from KNN Plugin
     */
    protected Response getKnnStats(List<String> nodeIds, List<String> stats) throws IOException {
        return executeKnnStatRequest(nodeIds, stats, KNNPlugin.KNN_BASE_URI);
    }

    protected Response executeKnnStatRequest(List<String> nodeIds, List<String> stats, final String baseURI) throws IOException {
        String nodePrefix = "";
        if (!nodeIds.isEmpty()) {
            nodePrefix = "/" + String.join(",", nodeIds);
        }

        String statsSuffix = "";
        if (!stats.isEmpty()) {
            statsSuffix = "/" + String.join(",", stats);
        }

        Request request = new Request("GET", baseURI + nodePrefix + "/stats" + statsSuffix);

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        return response;
    }

    /**
     * Warmup KNN Index
     */
    protected Response knnWarmup(List<String> indices) throws IOException {
        return executeWarmupRequest(indices, KNNPlugin.KNN_BASE_URI);
    }

    protected Response executeWarmupRequest(List<String> indices, final String baseURI) throws IOException {
        String indicesSuffix = "/" + String.join(",", indices);
        Request request = new Request("GET", baseURI + "/warmup" + indicesSuffix);
        return client().performRequest(request);
    }

    /**
     * Parse KNN Cluster stats from response
     */
    protected Map<String, Object> parseClusterStatsResponse(String responseBody) throws IOException {
        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();
        responseMap.remove("cluster_name");
        responseMap.remove("_nodes");
        responseMap.remove("nodes");
        return responseMap;
    }

    /**
     * Parse KNN Node stats from response
     */
    protected List<Map<String, Object>> parseNodeStatsResponse(String responseBody) throws IOException {
        @SuppressWarnings("unchecked")
        Map<String, Object> responseMap = (Map<String, Object>) createParser(XContentType.JSON.xContent(), responseBody).map().get("nodes");

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nodeResponses = responseMap.keySet()
            .stream()
            .map(key -> (Map<String, Object>) responseMap.get(key))
            .collect(Collectors.toList());

        return nodeResponses;
    }

    /**
     * Get the total hits from search response
     */
    @SuppressWarnings("unchecked")
    protected int parseTotalSearchHits(String searchResponseBody) throws IOException {
        Map<String, Object> responseMap = (Map<String, Object>) createParser(XContentType.JSON.xContent(), searchResponseBody).map()
            .get("hits");

        return (int) ((Map<String, Object>) responseMap.get("total")).get("value");
    }

    /**
     * Get the total number of graphs in the cache across all nodes
     */
    @SuppressWarnings("unchecked")
    protected int getTotalGraphsInCache() throws Exception {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());

        List<Map<String, Object>> nodesStats = parseNodeStatsResponse(responseBody);

        logger.info("[KNN] Node stats:  " + nodesStats);

        return nodesStats.stream()
            .filter(nodeStats -> nodeStats.get(INDICES_IN_CACHE.getName()) != null)
            .map(nodeStats -> nodeStats.get(INDICES_IN_CACHE.getName()))
            .mapToInt(
                nodeIndicesStats -> ((Map<String, Map<String, Object>>) nodeIndicesStats).values()
                    .stream()
                    .mapToInt(nodeIndexStats -> (int) nodeIndexStats.get(GRAPH_COUNT))
                    .sum()
            )
            .sum();
    }

    /**
     * Get specific Index setting value from response
     */
    protected String getIndexSettingByName(String indexName, String settingName) throws IOException {
        @SuppressWarnings("unchecked")
        Map<String, Object> settings = (Map<String, Object>) ((Map<String, Object>) getIndexSettings(indexName).get(indexName)).get(
            "settings"
        );
        return (String) settings.get(settingName);
    }

    protected void createModelSystemIndex() throws IOException {
        URL url = ModelDao.class.getClassLoader().getResource(MODEL_INDEX_MAPPING_PATH);
        if (url == null) {
            throw new IllegalStateException("Unable to retrieve mapping for \"" + MODEL_INDEX_NAME + "\"");
        }

        String mapping = Resources.toString(url, Charsets.UTF_8);
        mapping = mapping.substring(1, mapping.length() - 1);

        createIndex(MODEL_INDEX_NAME, Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).build(), mapping);
    }

    protected void addModelToSystemIndex(String modelId, ModelMetadata modelMetadata, byte[] model) throws IOException {
        assertFalse(Strings.isNullOrEmpty(modelId));
        String modelBase64 = Base64.getEncoder().encodeToString(model);

        Request request = new Request("POST", "/" + MODEL_INDEX_NAME + "/_doc/" + modelId + "?refresh=true");

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(MODEL_ID, modelId)
            .field(MODEL_STATE, modelMetadata.getState().getName())
            .field(KNN_ENGINE, modelMetadata.getKnnEngine().getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, modelMetadata.getSpaceType().getValue())
            .field(DIMENSION, modelMetadata.getDimension())
            .field(MODEL_BLOB_PARAMETER, modelBase64)
            .field(MODEL_TIMESTAMP, modelMetadata.getTimestamp())
            .field(MODEL_DESCRIPTION, modelMetadata.getDescription())
            .field(MODEL_ERROR, modelMetadata.getError())
            .endObject();

        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Clear cache
     * <p>
     * This function is a temporary workaround. Right now, we do not have a way of clearing the cache except by deleting
     * an index or changing k-NN settings. That being said, this function bounces a random k-NN setting in order to
     * clear the cache.
     */
    protected void clearCache() throws Exception {
        updateClusterSettings(KNNSettings.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES, "1m");
        updateClusterSettings(KNNSettings.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES, null);
    }

    /**
     * Clear script cache
     * <p>
     * Remove k-NN script from script cache so that it has to be recompiled
     */
    protected void clearScriptCache() throws Exception {
        updateClusterSettings("script.context.score.cache_expire", "0");
        updateClusterSettings("script.context.score.cache_expire", null);
    }

    private Script buildScript(String source, String language, Map<String, Object> params) {
        return new Script(Script.DEFAULT_SCRIPT_TYPE, language, source, params);
    }

    private ScriptedMetricAggregationBuilder getScriptedMetricAggregationBuilder(
        String initScriptSource,
        String mapScriptSource,
        String combineScriptSource,
        String reduceScriptSource,
        String language,
        String aggName
    ) {
        String scriptLanguage = language != null ? language : Script.DEFAULT_SCRIPT_LANG;
        Script initScript = buildScript(initScriptSource, scriptLanguage, Collections.emptyMap());
        Script mapScript = buildScript(mapScriptSource, scriptLanguage, Collections.emptyMap());
        Script combineScript = buildScript(combineScriptSource, scriptLanguage, Collections.emptyMap());
        Script reduceScript = buildScript(reduceScriptSource, scriptLanguage, Collections.emptyMap());
        return new ScriptedMetricAggregationBuilder(aggName).mapScript(mapScript)
            .combineScript(combineScript)
            .reduceScript(reduceScript)
            .initScript(initScript);
    }

    protected Request constructScriptedMetricAggregationSearchRequest(
        String aggName,
        String language,
        String initScriptSource,
        String mapScriptSource,
        String combineScriptSource,
        String reduceScriptSource,
        int size
    ) throws Exception {

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field("size", size).startObject("query");
        builder.startObject("match_all");
        builder.endObject();
        builder.endObject();
        builder.startObject("aggs");
        final ScriptedMetricAggregationBuilder scriptedMetricAggregationBuilder = getScriptedMetricAggregationBuilder(
            initScriptSource,
            mapScriptSource,
            combineScriptSource,
            reduceScriptSource,
            language,
            aggName
        );
        scriptedMetricAggregationBuilder.toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject();
        builder.endObject();
        String endpoint = String.format(Locale.getDefault(), "/%s/_search?size=0&filter_path=aggregations", INDEX_NAME);
        Request request = new Request("POST", endpoint);
        request.setJsonEntity(Strings.toString(builder));
        return request;
    }

    protected Request constructScriptScoreContextSearchRequest(
        String indexName,
        QueryBuilder qb,
        Map<String, Object> params,
        String language,
        String source,
        int size
    ) throws Exception {
        Script script = buildScript(source, language, params);
        ScriptScoreQueryBuilder sc = new ScriptScoreQueryBuilder(qb, script);
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field("size", size).startObject("query");
        builder.startObject("script_score");
        builder.field("query");
        sc.query().toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.field("script", script);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        Request request = new Request("POST", "/" + indexName + "/_search");
        request.setJsonEntity(Strings.toString(builder));
        return request;
    }

    protected Request constructKNNScriptQueryRequest(String indexName, QueryBuilder qb, Map<String, Object> params) throws Exception {
        Script script = new Script(Script.DEFAULT_SCRIPT_TYPE, KNNScoringScriptEngine.NAME, KNNScoringScriptEngine.SCRIPT_SOURCE, params);
        ScriptScoreQueryBuilder sc = new ScriptScoreQueryBuilder(qb, script);
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        builder.startObject("script_score");
        builder.field("query");
        sc.query().toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.field("script", script);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        Request request = new Request("POST", "/" + indexName + "/_search");
        request.setJsonEntity(Strings.toString(builder));
        return request;
    }

    protected Request constructKNNScriptQueryRequest(String indexName, QueryBuilder qb, Map<String, Object> params, int size)
        throws Exception {
        return constructScriptScoreContextSearchRequest(
            indexName,
            qb,
            params,
            KNNScoringScriptEngine.NAME,
            KNNScoringScriptEngine.SCRIPT_SOURCE,
            size
        );
    }

    public Map<String, Object> xContentBuilderToMap(XContentBuilder xContentBuilder) {
        return XContentHelper.convertToMap(BytesReference.bytes(xContentBuilder), true, xContentBuilder.contentType()).v2();
    }

    public void bulkIngestRandomVectors(String indexName, String fieldName, int numVectors, int dimension) throws IOException {
        for (int i = 0; i < numVectors; i++) {
            float[] vector = new float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = randomFloat();
            }

            addKnnDoc(indexName, String.valueOf(i + 1), fieldName, Floats.asList(vector).toArray());
        }

    }

    // Method that adds multiple documents into the index using Bulk API
    public void bulkAddKnnDocs(String index, String fieldName, float[][] indexVectors, int docCount) throws IOException {
        Request request = new Request("POST", "/_bulk");

        request.addParameter("refresh", "true");
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < docCount; i++) {
            sb.append("{ \"index\" : { \"_index\" : \"")
                .append(index)
                .append("\", \"_id\" : \"")
                .append(i)
                .append("\" } }\n")
                .append("{ \"")
                .append(fieldName)
                .append("\" : ")
                .append(Arrays.toString(indexVectors[i]))
                .append(" }\n");
        }

        request.setJsonEntity(sb.toString());

        Response response = client().performRequest(request);
        assertEquals(response.getStatusLine().getStatusCode(), 200);
    }

    // Method that returns index vectors of the documents that were added before into the index
    public float[][] getIndexVectorsFromIndex(String testIndex, String testField, int docCount, int dimensions) throws Exception {
        float[][] vectors = new float[docCount][dimensions];

        QueryBuilder qb = new MatchAllQueryBuilder();

        Request request = new Request("POST", "/" + testIndex + "/_search");

        request.addParameter("size", Integer.toString(docCount));
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder.field("query", qb);
        builder.endObject();
        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), testField);
        int i = 0;

        for (KNNResult result : results) {
            float[] primitiveArray = Floats.toArray(Arrays.stream(result.getVector()).collect(Collectors.toList()));
            vectors[i++] = primitiveArray;
        }

        return vectors;
    }

    // Method that performs bulk search for multiple queries and stores the resulting documents ids into list
    public List<List<String>> bulkSearch(String testIndex, String testField, float[][] queryVectors, int k) throws Exception {
        List<List<String>> searchResults = new ArrayList<>();
        List<String> kVectors;

        for (int i = 0; i < queryVectors.length; i++) {
            KNNQueryBuilder knnQueryBuilderRecall = new KNNQueryBuilder(testField, queryVectors[i], k);
            Response respRecall = searchKNNIndex(testIndex, knnQueryBuilderRecall, k);
            List<KNNResult> resultsRecall = parseSearchResponse(EntityUtils.toString(respRecall.getEntity()), testField);

            assertEquals(resultsRecall.size(), k);
            kVectors = new ArrayList<>();
            for (KNNResult result : resultsRecall) {
                kVectors.add(result.getDocId());
            }
            searchResults.add(kVectors);
        }

        return searchResults;
    }

    // Method that waits till the health of nodes in the cluster goes green
    public void waitForClusterHealthGreen(String numOfNodes) throws IOException {
        Request waitForGreen = new Request("GET", "/_cluster/health");
        waitForGreen.addParameter("wait_for_nodes", numOfNodes);
        waitForGreen.addParameter("wait_for_status", "green");
        client().performRequest(waitForGreen);
    }

    // Add KNN docs into a KNN index by providing the initial documentID and number of documents
    public void addKNNDocs(String testIndex, String testField, int dimension, int firstDocID, int numDocs) throws IOException {
        for (int i = firstDocID; i < firstDocID + numDocs; i++) {
            Float[] indexVector = new Float[dimension];
            Arrays.fill(indexVector, (float) i);
            addKnnDoc(testIndex, Integer.toString(i), testField, indexVector);
        }
    }

    // Validate KNN search on a KNN index by generating the query vector from the number of documents in the index
    public void validateKNNSearch(String testIndex, String testField, int dimension, int numDocs, int k) throws Exception {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);

        Response searchResponse = searchKNNIndex(testIndex, new KNNQueryBuilder(testField, queryVector, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), testField);

        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
        }
    }

    protected Settings createKNNIndexCustomLegacyFieldMappingSettings(SpaceType spaceType, Integer m, Integer ef_construction) {
        return Settings.builder()
            .put(NUMBER_OF_SHARDS, 1)
            .put(NUMBER_OF_REPLICAS, 0)
            .put(INDEX_KNN, true)
            .put(KNNSettings.KNN_SPACE_TYPE, spaceType.getValue())
            .put(KNNSettings.KNN_ALGO_PARAM_M, m)
            .put(KNNSettings.KNN_ALGO_PARAM_EF_CONSTRUCTION, ef_construction)
            .build();
    }

    public String createKNNIndexMethodFieldMapping(String fieldName, Integer dimensions) throws IOException {
        return Strings.toString(
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(fieldName)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, dimensions.toString())
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
        );
    }

    public String createKNNIndexCustomMethodFieldMapping(
        String fieldName,
        Integer dimensions,
        SpaceType spaceType,
        String engine,
        Integer m,
        Integer ef_construction
    ) throws IOException {
        return Strings.toString(
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(fieldName)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, dimensions.toString())
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .field(KNN_ENGINE, engine)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, ef_construction)
                .field(METHOD_PARAMETER_M, m)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
        );
    }

    // Default KNN script score settings
    protected Settings createKNNDefaultScriptScoreSettings() {
        return Settings.builder().put(NUMBER_OF_SHARDS, 1).put(NUMBER_OF_REPLICAS, 0).put(INDEX_KNN, false).build();
    }

    // Validate script score search for these space_types : {"l2", "l1", "linf"}
    protected void validateKNNScriptScoreSearch(String testIndex, String testField, int dimension, int numDocs, int k, SpaceType spaceType)
        throws Exception {

        IDVectorProducer idVectorProducer = new IDVectorProducer(dimension, numDocs);
        float[] queryVector = idVectorProducer.getVector(numDocs);

        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        params.put(FIELD, testField);
        params.put(QUERY_VALUE, queryVector);
        params.put(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());

        Request request = constructKNNScriptQueryRequest(testIndex, qb, params, k);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), testField);
        assertEquals(k, results.size());

        PriorityQueue<DistVector> pq = computeGroundTruthValues(k, spaceType, idVectorProducer);

        for (int i = k - 1; i >= 0; i--) {
            int expDocID = Integer.parseInt(pq.poll().getDocID());
            int actualDocID = Integer.parseInt(results.get(i).getDocId());
            assertEquals(expDocID, actualDocID);
        }
    }

    // validate KNN painless script score search for the space_types : "l2", "l1"
    protected void validateKNNPainlessScriptScoreSearch(String testIndex, String testField, String source, int numDocs, int k)
        throws Exception {
        QueryBuilder qb = new MatchAllQueryBuilder();
        Request request = constructScriptScoreContextSearchRequest(
            testIndex,
            qb,
            Collections.emptyMap(),
            Script.DEFAULT_SCRIPT_LANG,
            source,
            k
        );
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), testField);
        assertEquals(k, results.size());

        for (int i = 0; i < k; i++) {
            int expDocID = numDocs - i - 1;
            int actualDocID = Integer.parseInt(results.get(i).getDocId());
            assertEquals(expDocID, actualDocID);
        }
    }

    // create painless script source for space_type "l2" by creating query vector based on number of documents
    protected String createL2PainlessScriptSource(String testField, int dimension, int numDocs) {
        IDVectorProducer idVectorProducer = new IDVectorProducer(dimension, numDocs);
        float[] queryVector = idVectorProducer.getVector(numDocs);
        return String.format("1/(1 + l2Squared(" + Arrays.toString(queryVector) + ", doc['%s']))", testField);
    }

    // create painless script source for space_type "l1" by creating query vector based on number of documents
    protected String createL1PainlessScriptSource(String testField, int dimension, int numDocs) {
        IDVectorProducer idVectorProducer = new IDVectorProducer(dimension, numDocs);
        float[] queryVector = idVectorProducer.getVector(numDocs);
        return String.format("1/(1 + l1Norm(" + Arrays.toString(queryVector) + ", doc['%s']))", testField);
    }

    /**
     * Method that call train api and produces a trained model
     *
     * @param modelId to identify the model. If null, one will be autogenerated
     * @param trainingIndexName index to pull training data from
     * @param trainingFieldName field to pull training data from
     * @param dimension dimension of model
     * @param method method definition for model
     * @param description description of model
     * @return Response returned by the cluster
     * @throws IOException if request cannot be performed
     */
    public Response trainModel(
        String modelId,
        String trainingIndexName,
        String trainingFieldName,
        int dimension,
        Map<String, Object> method,
        String description
    ) throws IOException {

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, trainingIndexName)
            .field(TRAIN_FIELD_PARAMETER, trainingFieldName)
            .field(DIMENSION, dimension)
            .field(KNN_METHOD, method)
            .field(MODEL_DESCRIPTION, description)
            .endObject();

        if (modelId == null) {
            modelId = "";
        } else {
            modelId = "/" + modelId;
        }

        Request request = new Request("POST", "/_plugins/_knn/models" + modelId + "/_train");
        request.setJsonEntity(Strings.toString(builder));
        return client().performRequest(request);
    }

    /**
     * Retrieve the model
     *
     * @param modelId Id of model to be retrieved
     * @param filters filters to filter fields out. If null, no filters will
     * @return Response from cluster
     * @throws IOException if request cannot be performed
     */
    public Response getModel(String modelId, List<String> filters) throws IOException {

        if (modelId == null) {
            modelId = "";
        } else {
            modelId = "/" + modelId;
        }

        String filterString = "";

        if (filters != null && !filters.isEmpty()) {
            filterString = "&filter_path=" + StringUtils.join(filters, ",");
        }

        Request request = new Request("GET", "/_plugins/_knn/models" + modelId + filterString);

        return client().performRequest(request);
    }

    public void assertTrainingSucceeds(String modelId, int attempts, int delayInMillis) throws InterruptedException, Exception {
        int attemptNum = 0;
        Response response;
        Map<String, Object> responseMap;
        ModelState modelState;
        while (attemptNum < attempts) {
            Thread.sleep(delayInMillis);
            attemptNum++;

            response = getModel(modelId, null);

            responseMap = createParser(XContentType.JSON.xContent(), EntityUtils.toString(response.getEntity())).map();

            modelState = ModelState.getModelState((String) responseMap.get(MODEL_STATE));
            if (modelState == ModelState.CREATED) {
                return;
            }

            assertNotEquals(ModelState.FAILED, modelState);
        }

        fail("Training did not succeed after " + attempts + " attempts with a delay of " + delayInMillis + " ms.");
    }

    public void assertTrainingFails(String modelId, int attempts, int delayInMillis) throws Exception {
        int attemptNum = 0;
        Response response;
        Map<String, Object> responseMap;
        ModelState modelState;
        while (attemptNum < attempts) {
            Thread.sleep(delayInMillis);
            attemptNum++;

            response = getModel(modelId, null);

            responseMap = createParser(XContentType.JSON.xContent(), EntityUtils.toString(response.getEntity())).map();

            modelState = ModelState.getModelState((String) responseMap.get(MODEL_STATE));
            if (modelState == ModelState.FAILED) {
                return;
            }

            assertNotEquals(ModelState.CREATED, modelState);
        }

        fail("Training did not succeed after " + attempts + " attempts with a delay of " + delayInMillis + " ms.");
    }

    /**
     * We need to be able to dump the jacoco coverage before cluster is shut down.
     * The new internal testing framework removed some of the gradle tasks we were listening to
     * to choose a good time to do it. This will dump the executionData to file after each test.
     * TODO: This is also currently just overwriting integTest.exec with the updated execData without
     * resetting after writing each time. This can be improved to either write an exec file per test
     * or by letting jacoco append to the file
     */
    public interface IProxy {
        byte[] getExecutionData(boolean reset);

        void dump(boolean reset);

        void reset();
    }
}
