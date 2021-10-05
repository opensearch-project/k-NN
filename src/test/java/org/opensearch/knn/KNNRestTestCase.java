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

package org.opensearch.knn;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNQueryBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.script.KNNScoringScriptEngine;
import org.apache.http.util.EntityUtils;
import org.junit.AfterClass;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.Strings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.index.query.ExistsQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreQueryBuilder;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.RestStatus;
import org.opensearch.script.Script;

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
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ERROR;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_MAPPING_PATH;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.common.KNNConstants.MODEL_TIMESTAMP;
import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.GRAPH_COUNT;
import static org.opensearch.knn.plugin.stats.StatNames.INDICES_IN_CACHE;

/**
 * Base class for integration tests for KNN plugin. Contains several methods for testing KNN ES functionality.
 */
public class KNNRestTestCase extends ODFERestTestCase {
    public static final String INDEX_NAME = "test_index";
    public static final String FIELD_NAME = "test_field";

    @AfterClass
    public static void dumpCoverage() throws IOException, MalformedObjectNameException {
        // jacoco.dir is set in esplugin-coverage.gradle, if it doesn't exist we don't
        // want to collect coverage so we can return early
        String jacocoBuildPath = System.getProperty("jacoco.dir");
        if (Strings.isNullOrEmpty(jacocoBuildPath)) {
            return;
        }

        String serverUrl = "service:jmx:rmi:///jndi/rmi://127.0.0.1:7777/jmxrmi";
        try (JMXConnector connector = JMXConnectorFactory.connect(new JMXServiceURL(serverUrl))) {
            IProxy proxy = MBeanServerInvocationHandler.newProxyInstance(
                connector.getMBeanServerConnection(), new ObjectName("org.jacoco:type=Runtime"), IProxy.class,
                false);

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

    /**
     * Run KNN Search on Index
     */
    protected Response searchKNNIndex(String index, KNNQueryBuilder knnQueryBuilder, int resultSize) throws
        IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnQueryBuilder.doXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject().endObject();

        Request request = new Request(
            "POST",
            "/" + index + "/_search"
        );

        request.addParameter("size", Integer.toString(resultSize));
        request.addParameter("explain", Boolean.toString(true));
        request.addParameter("search_type", "query_then_fetch");
        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }

    /**
     * Run exists search
     */
    protected Response searchExists(String index, ExistsQueryBuilder existsQueryBuilder, int resultSize) throws
        IOException {

        Request request = new Request(
            "POST",
            "/" + index + "/_search"
        );

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        builder = XContentFactory.jsonBuilder().startObject();
        builder.field("query", existsQueryBuilder);
        builder.endObject();

        request.addParameter("size", Integer.toString(resultSize));
        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }

    /**
     * Parse the response of KNN search into a List of KNNResults
     */
    protected List<KNNResult> parseSearchResponse(String responseBody, String fieldName) throws IOException {
        @SuppressWarnings("unchecked")
        List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(XContentType.JSON.xContent(),
            responseBody).map().get("hits")).get("hits");

        @SuppressWarnings("unchecked")
        List<KNNResult> knnSearchResponses = hits.stream().map(hit -> {
                @SuppressWarnings("unchecked")
                Float[] vector = Arrays.stream(
                    ((ArrayList<Float>) ((Map<String, Object>)
                        ((Map<String, Object>) hit).get("_source")).get(fieldName)).toArray())
                    .map(Object::toString)
                    .map(Float::valueOf)
                    .toArray(Float[]::new);
                return new KNNResult((String) ((Map<String, Object>) hit).get("_id"), vector);
            }
        ).collect(Collectors.toList());

        return knnSearchResponses;
    }
    protected List<Float> parseSearchResponseScore(String responseBody, String fieldName) throws IOException {
        @SuppressWarnings("unchecked")
        List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(XContentType.JSON.xContent(),
                responseBody).map().get("hits")).get("hits");

        @SuppressWarnings("unchecked")
        List<Float> knnSearchResponses = hits.stream().map(hit ->
                ((Double) ((Map<String, Object>) hit).get("_score")).floatValue()).collect(Collectors.toList());

        return knnSearchResponses;
    }

    /**
     * Parse the score from the KNN search response
     */

    /**
     * Delete KNN index
     */
    protected void deleteKNNIndex(String index) throws IOException {
        Request request = new Request(
            "DELETE",
            "/" + index
        );

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * For a given index, make a mapping request
     */
    protected void putMappingRequest(String index, String mapping) throws IOException {
        // Put KNN mapping
        Request request = new Request(
            "PUT",
            "/" + index + "/_mapping"
        );

        request.setJsonEntity(mapping);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Utility to create a Knn Index Mapping
     */
    protected String createKnnIndexMapping(String fieldName, Integer dimensions) throws IOException {
        return Strings.toString(XContentFactory.jsonBuilder().startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimensions.toString())
            .endObject()
            .endObject()
            .endObject());
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
    public Map<String, Object> getIndexMappingAsMap(String index) throws IOException {
        Request request = new Request(
                "GET",
                "/" + index + "/_mapping"
        );

        Response response = client().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
                RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();


        return (Map<String, Object>) ((Map<String, Object>) responseMap.get(index)).get("mappings");
    }

    public int getDocCount(String indexName) throws IOException {
        Request request = new Request(
          "GET",
          "/" + indexName + "/_count"
        );

        Response response = client().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
                RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();
        return (Integer) responseMap.get("count");
    }

    /**
     * Force merge KNN index segments
     */
    protected void forceMergeKnnIndex(String index) throws Exception {
        Request request = new Request(
            "POST",
            "/" + index + "/_refresh"
        );

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        request = new Request(
            "POST",
            "/" + index + "/_forcemerge"
        );

        request.addParameter("max_num_segments", "1");
        request.addParameter("flush", "true");
        response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        TimeUnit.SECONDS.sleep(5); // To make sure force merge is completed
    }

    /**
     * Add a single KNN Doc to an index
     */
    protected void addKnnDoc(String index, String docId, String fieldName, Object[] vector) throws IOException {
        Request request = new Request(
            "POST",
            "/" + index + "/_doc/" + docId + "?refresh=true"
        );

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject()
            .field(fieldName, vector)
            .endObject();
        request.setJsonEntity(Strings.toString(builder));
        client().performRequest(request);

        request = new Request(
            "POST",
            "/" + index + "/_refresh"
        );
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Add a single KNN Doc to an index with multiple fields
     */
    protected void addKnnDoc(String index, String docId, List<String> fieldNames, List<Object[]> vectors) throws IOException {
        Request request = new Request(
            "POST",
            "/" + index + "/_doc/" + docId + "?refresh=true"
        );

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        for (int i = 0; i < fieldNames.size(); i++) {
            builder.field(fieldNames.get(i), vectors.get(i));
        }
        builder.endObject();

        request.setJsonEntity(Strings.toString(builder));
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Add a single numeric field Doc to an index
     */
    protected void addDocWithNumericField(String index, String docId, String fieldName, long value) throws IOException {
        Request request = new Request(
            "POST",
            "/" + index + "/_doc/" + docId + "?refresh=true"
        );

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject()
            .field(fieldName, value)
            .endObject();

        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);


        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Add a single numeric field Doc to an index
     */
    protected void addDocWithBinaryField(String index, String docId, String fieldName, String base64String)
        throws IOException {
        Request request = new Request(
            "POST",
            "/" + index + "/_doc/" + docId + "?refresh=true"
        );

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject()
            .field(fieldName, base64String)
            .endObject();

        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);

        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Update a KNN Doc with a new vector for the given fieldName
     */
    protected void updateKnnDoc(String index, String docId, String fieldName, Object[] vector) throws IOException {
        Request request = new Request(
            "POST",
            "/" + index + "/_doc/" + docId + "?refresh=true"
        );

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject()
            .field(fieldName, vector)
            .endObject();

        request.setJsonEntity(Strings.toString(builder));

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Delete Knn Doc
     */
    protected void deleteKnnDoc(String index, String docId) throws IOException {
        // Put KNN mapping
        Request request = new Request(
            "DELETE",
            "/" + index + "/_doc/" + docId + "?refresh"
        );

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode()));
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
        return Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .build();
    }

    /**
     * Get Stats from KNN Plugin
     */
    protected Response getKnnStats(List<String> nodeIds, List<String> stats) throws IOException {
        return executeKnnStatRequest(nodeIds, stats, KNNPlugin.KNN_BASE_URI);
    }

    protected Response executeKnnStatRequest(
        List<String> nodeIds, List<String> stats, final String baseURI) throws IOException {
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
        Map<String, Object> responseMap = (Map<String, Object>) createParser(XContentType.JSON.xContent(),
            responseBody).map().get("nodes");

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nodeResponses = responseMap.keySet().stream().map(key ->
            (Map<String, Object>) responseMap.get(key)
        ).collect(Collectors.toList());

        return nodeResponses;
    }

    /**
     * Get the total hits from search response
     */
    @SuppressWarnings("unchecked")
    protected int parseTotalSearchHits(String searchResponseBody) throws IOException {
        Map<String, Object> responseMap = (Map<String, Object>) createParser(
            XContentType.JSON.xContent(),
            searchResponseBody
        ).map().get("hits");

        return (int) ((Map<String, Object>) responseMap.get("total")).get("value");
    }

    /**
     * Get the total number of graphs in the cache across all nodes
     */
    @SuppressWarnings("unchecked")
    protected int getTotalGraphsInCache() throws IOException {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());

        List<Map<String, Object>> nodesStats = parseNodeStatsResponse(responseBody);

        logger.info("[KNN] Node stats:  " + nodesStats);

        return nodesStats.stream()
            .filter(nodeStats -> nodeStats.get(INDICES_IN_CACHE.getName()) != null)
            .map(nodeStats -> nodeStats.get(INDICES_IN_CACHE.getName()))
            .mapToInt(nodeIndicesStats ->
                ((Map<String, Map<String, Object>>) nodeIndicesStats).values().stream()
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
        Map<String, Object> settings =
            (Map<String, Object>) ((Map<String, Object>) getIndexSettings(indexName).get(indexName))
                .get("settings");
        return (String) settings.get(settingName);
    }

    protected void createModelSystemIndex() throws IOException {
        URL url = ModelDao.class.getClassLoader().getResource(MODEL_INDEX_MAPPING_PATH);
        if (url == null) {
            throw new IllegalStateException("Unable to retrieve mapping for \"" + MODEL_INDEX_NAME + "\"");
        }

        String mapping = Resources.toString(url, Charsets.UTF_8);
        mapping = mapping.substring(1, mapping.length() - 1);

        createIndex(MODEL_INDEX_NAME, Settings.builder()
                .put("number_of_shards", 1)
                .put("number_of_replicas", 0).build(),
                mapping);
    }

    protected void addModelToSystemIndex(String modelId, ModelMetadata modelMetadata, byte[] model) throws IOException {
        assertFalse(Strings.isNullOrEmpty(modelId));
        String modelBase64 = Base64.getEncoder().encodeToString(model);

        Request request = new Request(
                "POST",
                "/" + MODEL_INDEX_NAME + "/_doc/" + modelId + "?refresh=true"
        );

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject()
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

        assertEquals(request.getEndpoint() + ": failed", RestStatus.CREATED,
                RestStatus.fromCode(response.getStatusLine().getStatusCode()));
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

    protected Request constructScriptQueryRequest(
        String indexName, QueryBuilder qb, Map<String, Object> params, String language, String source, int size)
        throws Exception {
        Script script = new Script(Script.DEFAULT_SCRIPT_TYPE, language, source, params);
        ScriptScoreQueryBuilder sc = new ScriptScoreQueryBuilder(qb, script);
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject()
            .field("size", size)
            .startObject("query");
        builder.startObject("script_score");
        builder.field("query");
        sc.query().toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.field("script", script);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        Request request = new Request(
            "POST",
            "/" + indexName + "/_search"
        );
        request.setJsonEntity(Strings.toString(builder));
        return request;
    }

    protected Request constructKNNScriptQueryRequest(String indexName, QueryBuilder qb, Map<String, Object> params)
        throws Exception {
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
        Request request = new Request(
            "POST",
            "/" + indexName + "/_search"
        );
        request.setJsonEntity(Strings.toString(builder));
        return request;
    }

    protected Request constructKNNScriptQueryRequest(String indexName, QueryBuilder qb, Map<String, Object> params,
                                                     int size) throws Exception {
        return constructScriptQueryRequest(indexName, qb, params, KNNScoringScriptEngine.NAME, KNNScoringScriptEngine.SCRIPT_SOURCE, size);
    }

    public Map<String, Object> xContentBuilderToMap(XContentBuilder xContentBuilder) {
        return XContentHelper.convertToMap(BytesReference.bytes(xContentBuilder), true,
                xContentBuilder.contentType()).v2();
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
