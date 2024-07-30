/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.script.Script;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class PainlessScriptIT extends KNNRestTestCase {

    public static final int AGGREGATION_FIELD_NAME_MIN_LENGTH = 2;
    public static final int AGGREGATION_FIELD_NAME_MAX_LENGTH = 5;
    private static final String NUMERIC_INDEX_FIELD_NAME = "price";

    /**
     * Utility to create a Index Mapping with multiple fields
     */
    protected String createMapping(List<MappingProperty> properties) throws IOException {
        Objects.requireNonNull(properties);
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().startObject("properties");
        for (MappingProperty property : properties) {
            XContentBuilder builder = xContentBuilder.startObject(property.getName()).field("type", property.getType());
            if (property.getDimension() != null) {
                builder.field("dimension", property.getDimension());
            }

            if (property.getDocValues() != null) {
                builder.field("doc_values", property.getDocValues());
            }

            if (property.getKnnMethodContext() != null) {
                builder.startObject(KNNConstants.KNN_METHOD);
                property.getKnnMethodContext().toXContent(builder, ToXContent.EMPTY_PARAMS);
                builder.endObject();
            }

            builder.endObject();
        }
        xContentBuilder.endObject().endObject();
        return xContentBuilder.toString();
    }

    /*
     creates KnnIndex based on properties, we add single non-knn vector documents to verify whether actions
     works on non-knn vector documents as well
     */
    private void buildTestIndex(Map<String, Float[]> knnDocuments) throws Exception {
        List<MappingProperty> properties = buildMappingProperties();
        buildTestIndex(knnDocuments, properties);
    }

    private void buildTestIndex(Map<String, Float[]> knnDocuments, List<MappingProperty> properties) throws Exception {
        createKnnIndex(INDEX_NAME, createMapping(properties));
        for (Map.Entry<String, Float[]> data : knnDocuments.entrySet()) {
            addKnnDoc(INDEX_NAME, data.getKey(), FIELD_NAME, data.getValue());
        }
    }

    private Map<String, Float[]> getKnnVectorTestData() {
        Map<String, Float[]> data = new HashMap<>();
        data.put("1", new Float[] { 100.0f, 1.0f });
        data.put("2", new Float[] { 99.0f, 2.0f });
        data.put("3", new Float[] { 97.0f, 3.0f });
        data.put("4", new Float[] { 98.0f, 4.0f });
        return data;
    }

    private Map<String, Float[]> getL2TestData() {
        Map<String, Float[]> data = new HashMap<>();
        data.put("1", new Float[] { 6.0f, 6.0f });
        data.put("2", new Float[] { 2.0f, 2.0f });
        data.put("3", new Float[] { 4.0f, 4.0f });
        data.put("4", new Float[] { 3.0f, 3.0f });
        return data;
    }

    private Map<String, Float[]> getL1TestData() {
        Map<String, Float[]> data = new HashMap<>();
        data.put("1", new Float[] { 6.0f, 6.0f });
        data.put("2", new Float[] { 4.0f, 1.0f });
        data.put("3", new Float[] { 3.0f, 3.0f });
        data.put("4", new Float[] { 5.0f, 5.0f });
        return data;
    }

    private Map<String, Float[]> getLInfTestData() {
        Map<String, Float[]> data = new HashMap<>();
        data.put("1", new Float[] { 6.0f, 6.0f });
        data.put("2", new Float[] { 4.0f, 1.0f });
        data.put("3", new Float[] { 3.0f, 3.0f });
        data.put("4", new Float[] { 5.0f, 5.0f });
        return data;
    }

    private Map<String, Float[]> getInnerProdTestData() {
        Map<String, Float[]> data = new HashMap<>();
        data.put("1", new Float[] { -2.0f, -2.0f });
        data.put("2", new Float[] { 1.0f, 1.0f });
        data.put("3", new Float[] { 2.0f, 2.0f });
        data.put("4", new Float[] { 2.0f, -2.0f });
        return data;
    }

    private Map<String, Float[]> getCosineTestData() {
        Map<String, Float[]> data = new HashMap<>();
        data.put("0", new Float[] { 1.0f, -1.0f });
        data.put("2", new Float[] { 1.0f, 1.0f });
        data.put("1", new Float[] { 1.0f, 0.0f });
        return data;
    }

    /*
     The doc['field'] will throw an error if field is missing from the mappings.
     */
    private List<MappingProperty> buildMappingProperties() {
        List<MappingProperty> properties = new ArrayList<>();
        properties.add(new MappingProperty(FIELD_NAME, KNNVectorFieldMapper.CONTENT_TYPE).dimension("2"));
        properties.add(new MappingProperty(NUMERIC_INDEX_FIELD_NAME, "integer"));
        return properties;
    }

    public void testL2ScriptScoreFails() throws Exception {
        String source = String.format("1/(1 + l2Squared([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getL2TestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
        deleteKNNIndex(INDEX_NAME);
    }

    private Request buildPainlessScoreScriptRequest(String source, int size, Map<String, Float[]> documents) throws Exception {
        buildTestIndex(documents);
        QueryBuilder qb = new MatchAllQueryBuilder();
        return constructScriptScoreContextSearchRequest(
            INDEX_NAME,
            qb,
            Collections.emptyMap(),
            Script.DEFAULT_SCRIPT_LANG,
            source,
            size,
            Collections.emptyMap()
        );
    }

    private Request buildPainlessScoreScriptRequest(
        String source,
        int size,
        Map<String, Float[]> documents,
        List<MappingProperty> properties
    ) throws Exception {
        buildTestIndex(documents, properties);
        QueryBuilder qb = new MatchAllQueryBuilder();
        return constructScriptScoreContextSearchRequest(
            INDEX_NAME,
            qb,
            Collections.emptyMap(),
            Script.DEFAULT_SCRIPT_LANG,
            source,
            size,
            Collections.emptyMap()
        );
    }

    private Request buildPainlessScriptedMetricRequest(
        String initScriptSource,
        String mapScriptSource,
        String combineScriptSource,
        String reduceScriptSource,
        Map<String, Float[]> documents,
        String aggName
    ) throws Exception {
        buildTestIndex(documents);
        return constructScriptedMetricAggregationSearchRequest(
            aggName,
            Script.DEFAULT_SCRIPT_LANG,
            initScriptSource,
            mapScriptSource,
            combineScriptSource,
            reduceScriptSource,
            documents.size()
        );
    }

    public void testL2ScriptScore() throws Exception {

        String source = String.format("1/(1 + l2Squared([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getL2TestData());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "2", "4", "3", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testGetValueReturnsDocValues() throws Exception {

        String source = String.format("doc['%s'].value[0]", FIELD_NAME);
        Map<String, Float[]> testData = getKnnVectorTestData();
        Request request = buildPainlessScoreScriptRequest(source, testData.size(), testData);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(testData.size(), results.size());

        String[] expectedDocIDs = { "1", "2", "4", "3" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testGetValueScriptFailsWithMissingField() throws Exception {
        String source = String.format("doc['%s']", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getKnnVectorTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
        deleteKNNIndex(INDEX_NAME);
    }

    public void testGetValueScriptFailsWithOutOfBoundException() throws Exception {
        Map<String, Float[]> testData = getKnnVectorTestData();
        String source = String.format("doc['%s'].value[%d]", FIELD_NAME, testData.get("1").length);
        Request request = buildPainlessScoreScriptRequest(source, testData.size(), testData);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
        deleteKNNIndex(INDEX_NAME);
    }

    public void testGetValueScriptScoreWithNumericField() throws Exception {

        String source = String.format("doc['%s'].size() == 0 ? 0 : doc['%s'].value[0]", FIELD_NAME, FIELD_NAME);
        Map<String, Float[]> testData = getKnnVectorTestData();
        Request request = buildPainlessScoreScriptRequest(source, testData.size(), testData);
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(testData.size(), results.size());

        String[] expectedDocIDs = { "1", "2", "4", "3" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testL2ScriptScoreWithNumericField() throws Exception {

        String source = String.format("doc['%s'].size() == 0 ? 0 : 1/(1 + l2Squared([1.0f, 1.0f], doc['%s']))", FIELD_NAME, FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getL2TestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "2", "4", "3", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testCosineSimilarityScriptScoreFails() throws Exception {
        String source = String.format("1 + cosineSimilarity([2.0f, -2.0f], doc['%s'])", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getCosineTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
        deleteKNNIndex(INDEX_NAME);
    }

    public void testCosineSimilarityScriptScore() throws Exception {
        String source = String.format("1 + cosineSimilarity([2.0f, -2.0f], doc['%s'])", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getCosineTestData());
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "0", "1", "2" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testCosineSimilarityScriptScoreWithNumericField() throws Exception {
        String source = String.format("doc['%s'].size() == 0 ? 0 : 1 + cosineSimilarity([2.0f, -2.0f], doc['%s'])", FIELD_NAME, FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getCosineTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "0", "1", "2" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    // test fails without size check before executing method
    public void testCosineSimilarityNormalizedScriptScoreFails() throws Exception {
        String source = String.format("1 + cosineSimilarity([2.0f, -2.0f], doc['%s'], 3.0f)", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getCosineTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
        deleteKNNIndex(INDEX_NAME);
    }

    public void testCosineSimilarityNormalizedScriptScore() throws Exception {
        String source = String.format("1 + cosineSimilarity([2.0f, -2.0f], doc['%s'], 3.0f)", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getCosineTestData());
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "0", "1", "2" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testCosineSimilarityNormalizedScriptScoreWithNumericField() throws Exception {
        String source = String.format(
            "doc['%s'].size() == 0 ? 0 : 1 + cosineSimilarity([2.0f, -2.0f], doc['%s'], 3.0f)",
            FIELD_NAME,
            FIELD_NAME
        );
        Request request = buildPainlessScoreScriptRequest(source, 3, getCosineTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "0", "1", "2" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    // L1 tests
    public void testL1ScriptScoreFails() throws Exception {
        String source = String.format("1/(1 + l1Norm([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getL1TestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
        deleteKNNIndex(INDEX_NAME);
    }

    public void testL1ScriptScore() throws Exception {

        String source = String.format("1/(1 + l1Norm([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getL1TestData());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "2", "3", "4", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testL1ScriptScoreWithNumericField() throws Exception {

        String source = String.format("doc['%s'].size() == 0 ? 0 : 1/(1 + l1Norm([1.0f, 1.0f], doc['%s']))", FIELD_NAME, FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getL1TestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "2", "3", "4", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    // L-inf tests
    public void testLInfScriptScoreFails() throws Exception {
        String source = String.format("1/(1 + lInfNorm([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getLInfTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
        deleteKNNIndex(INDEX_NAME);
    }

    public void testLInfScriptScore() throws Exception {

        String source = String.format("1/(1 + lInfNorm([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getLInfTestData());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "3", "2", "4", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testLInfScriptScoreWithNumericField() throws Exception {

        String source = String.format("doc['%s'].size() == 0 ? 0 : 1/(1 + lInfNorm([1.0f, 1.0f], doc['%s']))", FIELD_NAME, FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getLInfTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "3", "2", "4", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testInnerProdScriptScoreFails() throws Exception {
        String source = String.format("float x = innerProduct([1.0f, 1.0f], doc['%s']); return x >= 0? 2-1/(x+1):1/(1-x);", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getInnerProdTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
        deleteKNNIndex(INDEX_NAME);
    }

    public void testInnerProdScriptScore() throws Exception {

        String source = String.format("float x = innerProduct([1.0f, 1.0f], doc['%s']); return x >= 0? 2-1/(x+1):1/(1-x);", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getInnerProdTestData());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "3", "2", "4", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testInnerProdScriptScoreWithNumericField() throws Exception {

        String source = String.format(
            "if (doc['%s'].size() == 0) "
                + "{ return 0; } "
                + "else "
                + "{ float x = innerProduct([1.0f, 1.0f], doc['%s']); return x >= 0? 2-1/(x+1):1/(1-x); }",
            FIELD_NAME,
            FIELD_NAME
        );
        Request request = buildPainlessScoreScriptRequest(source, 3, getInnerProdTestData());
        addDocWithNumericField(INDEX_NAME, "100", NUMERIC_INDEX_FIELD_NAME, 1000);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "3", "2", "4", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    public void testScriptedMetricIsSupported() throws Exception {
        Map<String, Float[]> testData = getKnnVectorTestData();
        // sum of first value from each vector
        String initScriptSource = "state.x = []";
        String mapScriptSource = String.format("state.x.add(doc['%s'].value[0])", FIELD_NAME);
        String combineScriptSource = "double sum = 0; for (t in state.x) { sum += t } return sum";
        String reduceScriptSource = "double sum = 0; for (v in states) { sum += v } return sum";
        String aggName = randomAlphaOfLengthBetween(AGGREGATION_FIELD_NAME_MIN_LENGTH, AGGREGATION_FIELD_NAME_MAX_LENGTH); // random agg
                                                                                                                           // name for
                                                                                                                           // context
        Request request = buildPainlessScriptedMetricRequest(
            initScriptSource,
            mapScriptSource,
            combineScriptSource,
            reduceScriptSource,
            testData,
            aggName
        );
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        final String entity = EntityUtils.toString(response.getEntity());
        assertNotNull(entity);
        final double actualValue = parseAggregationResponse(entity, aggName);
        final double expectedSum = testData.values().stream().mapToDouble(vector -> vector[0]).sum();
        assertEquals("Script didn't produce sum of first dimension from all vectors", expectedSum, actualValue, 0.1);
        deleteKNNIndex(INDEX_NAME);
    }

    public void testL2ScriptingWithLuceneBackedIndex() throws Exception {
        List<MappingProperty> properties = new ArrayList<>();
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.DEFAULT,
            new MethodComponentContext(METHOD_HNSW, Collections.emptyMap())
        );
        properties.add(
            new MappingProperty(FIELD_NAME, KNNVectorFieldMapper.CONTENT_TYPE).dimension("2")
                .knnMethodContext(knnMethodContext)
                .docValues(randomBoolean())
        );

        String source = String.format("1/(1 + l2Squared([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Request request = buildPainlessScoreScriptRequest(source, 3, getL2TestData(), properties);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());

        String[] expectedDocIDs = { "2", "4", "3", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testHammingPainlessScript_whenBinary_thenSuccess() {
        int dimensions = 16;
        String mappingForKnnDisabled = createKnnIndexMapping(FIELD_NAME, dimensions, VectorDataType.BINARY);

        // 0b00000001, 0b00000001
        String source = String.format("1/(1 + hamming([1.0f, 1.0f], doc['%s']))", FIELD_NAME);

        Map<String, Float[]> data = new HashMap<>();
        data.put("1", new Float[] { (float) 0b00000001, (float) 0b00000001 });// Hamming distance 0
        data.put("2", new Float[] { (float) 0b01101111, (float) 0b00000010 });// Hamming distance 6
        data.put("3", new Float[] { (float) 0b01100010, (float) 0b00000011 });// Hamming distance 5
        data.put("4", new Float[] { (float) 0b00000001, (float) 0b01001100 });// Hamming distance 4

        Response response = buildIndexAndRunPainlessScript(source, 4, data, mappingForKnnDisabled, false);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(4, results.size());

        String[] expectedDocIDs = { "1", "4", "3", "2" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
    }

    @SneakyThrows
    public void testPainlessScript_whenNonBinary_thenException() {
        int dimensions = 2;
        String mapping = createKnnIndexMapping(FIELD_NAME, dimensions);
        String source = String.format("1/(1 + hamming([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Exception e = expectThrows(
            ResponseException.class,
            () -> buildIndexAndRunPainlessScript(source, 3, getKnnVectorTestData(), mapping, false)
        );
        assertTrue(e.getMessage(), e.getMessage().contains("The data type should be binary"));
    }

    @SneakyThrows
    public void testNonPainlessScript_whenBinary_thenException() {
        List<String> functions = Arrays.asList("l2Squared", "lInfNorm", "l1Norm", "innerProduct", "cosineSimilarity");
        int dimensions = 16;
        String mapping = createKnnIndexMapping(FIELD_NAME, dimensions, VectorDataType.BINARY);
        for (String function : functions) {
            String source = String.format(Locale.ROOT, "%s([1.0f, 1.0f], doc['%s'])", function, FIELD_NAME);
            Exception e = expectThrows(
                ResponseException.class,
                () -> buildIndexAndRunPainlessScript(source, 3, getKnnVectorTestData(), mapping, false)
            );
            assertTrue(e.getMessage(), e.getMessage().contains("The data type should be either float or byte"));
        }
    }

    private Response buildIndexAndRunPainlessScript(
        final String source,
        final int size,
        Map<String, Float[]> documents,
        final String mapper,
        final boolean enableKnn
    ) throws Exception {
        /*
         * Create knn index and populate data
         */
        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", enableKnn).build();
        createKnnIndex(INDEX_NAME, settings, mapper);
        try {
            for (Map.Entry<String, Float[]> data : documents.entrySet()) {
                addKnnDoc(INDEX_NAME, data.getKey(), FIELD_NAME, data.getValue());
            }
            QueryBuilder qb = new MatchAllQueryBuilder();
            Request request = constructScriptScoreContextSearchRequest(
                INDEX_NAME,
                qb,
                Collections.emptyMap(),
                Script.DEFAULT_SCRIPT_LANG,
                source,
                size,
                Collections.emptyMap()
            );
            return client().performRequest(request);
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    static class MappingProperty {

        private final String name;
        private final String type;
        private String dimension;

        private KNNMethodContext knnMethodContext;
        private Boolean docValues;

        MappingProperty(String name, String type) {
            this.name = name;
            this.type = type;
        }

        MappingProperty dimension(String dimension) {
            this.dimension = dimension;
            return this;
        }

        MappingProperty knnMethodContext(KNNMethodContext knnMethodContext) {
            this.knnMethodContext = knnMethodContext;
            return this;
        }

        MappingProperty docValues(boolean docValues) {
            this.docValues = docValues;
            return this;
        }

        KNNMethodContext getKnnMethodContext() {
            return knnMethodContext;
        }

        String getDimension() {
            return dimension;
        }

        String getName() {
            return name;
        }

        String getType() {
            return type;
        }

        Boolean getDocValues() {
            return docValues;
        }
    }
}
