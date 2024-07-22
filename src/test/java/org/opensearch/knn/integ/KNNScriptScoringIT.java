/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;

import lombok.SneakyThrows;
import org.opensearch.ExceptionsHelper;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreQueryBuilder;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringScriptEngine;
import org.opensearch.knn.plugin.script.KNNScoringSpace;
import org.opensearch.knn.plugin.script.KNNScoringSpaceFactory;
import org.opensearch.script.Script;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.hamcrest.Matchers.containsString;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;

public class KNNScriptScoringIT extends KNNRestTestCase {

    private static final String TEST_MODEL = "test-model";

    public void testKNNL2ScriptScore() throws Exception {
        testKNNScriptScore(SpaceType.L2);
    }

    public void testKNNL1ScriptScore() throws Exception {
        testKNNScriptScore(SpaceType.L1);
    }

    public void testKNNLInfScriptScore() throws Exception {
        testKNNScriptScore(SpaceType.LINF);
    }

    public void testKNNCosineScriptScore() throws Exception {
        testKNNScriptScore(SpaceType.COSINESIMIL);
    }

    @SneakyThrows
    public void testKNNHammingScriptScore() {
        testKNNScriptScoreOnBinaryIndex(SpaceType.HAMMING);
    }

    @SneakyThrows
    public void testKNNHammingScriptScore_whenNonBinary_thenException() {
        final int dims = randomIntBetween(2, 10) * 8;
        final float[] queryVector = randomVector(dims, VectorDataType.BYTE);
        final BiFunction<float[], float[], Float> scoreFunction = getScoreFunction(SpaceType.HAMMING, queryVector);
        List<VectorDataType> nonBinary = List.of(VectorDataType.FLOAT, VectorDataType.BYTE);
        for (VectorDataType vectorDataType : nonBinary) {
            Exception e = expectThrows(
                Exception.class,
                () -> createIndexAndAssertScriptScore(
                    createKnnIndexMapping(FIELD_NAME, dims, vectorDataType),
                    SpaceType.HAMMING,
                    scoreFunction,
                    dims,
                    queryVector,
                    true,
                    false,
                    vectorDataType
                )
            );
            assertTrue(e.getMessage(), e.getMessage().contains("data type should be [BINARY]"));
        }
    }

    public void testKNNNonHammingScriptScore_whenBinary_thenException() {
        final int dims = randomIntBetween(2, 10) * 8;
        final float[] queryVector = randomVector(dims, VectorDataType.BINARY);
        final BiFunction<float[], float[], Float> scoreFunction = getScoreFunction(SpaceType.HAMMING, queryVector);
        Set<SpaceType> spaceTypeToExclude = Set.of(SpaceType.UNDEFINED, SpaceType.HAMMING);
        Arrays.stream(SpaceType.values()).filter(s -> spaceTypeToExclude.contains(s) == false).forEach(s -> {
            Exception e = expectThrows(
                Exception.class,
                () -> createIndexAndAssertScriptScore(
                    createKnnIndexMapping(FIELD_NAME, dims, VectorDataType.BINARY),
                    s,
                    scoreFunction,
                    dims,
                    queryVector,
                    true,
                    false,
                    VectorDataType.BINARY
                )
            );
            assertTrue(e.getMessage(), e.getMessage().contains("Incompatible field_type"));
        });
    }

    public void testKNNInvalidSourceScript() throws Exception {
        /*
         * Create knn index and populate data
         */
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));

        /**
         * Construct Search Request
         */
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        /*
         *   params": {
         *       "field": "my_dense_vector",
         *       "query_value": [2.0, 2.0],
         *       "space_type": "cosinesimil"
         *      }
         */
        float[] queryVector = { 2.0f, -2.0f };
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.COSINESIMIL.getValue());
        Script script = new Script(Script.DEFAULT_SCRIPT_TYPE, KNNScoringScriptEngine.NAME, "Dummy_source", params);
        ScriptScoreQueryBuilder sc = new ScriptScoreQueryBuilder(qb, script);

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");

        builder.startObject("script_score");
        builder.field("query");
        sc.query().toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.field("script", script);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        Request request = new Request("POST", "/" + INDEX_NAME + "/_search");

        request.setJsonEntity(builder.toString());
        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString("Unknown script name Dummy_source"));
    }

    public void testInvalidSpace() throws Exception {
        String INVALID_SPACE = "dummy";
        /*
         * Create knn index and populate data
         */
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));

        /**
         * Construct Search Request
         */
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        float[] queryVector = { 2.0f, -2.0f };
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", INVALID_SPACE);
        Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertThat(
            EntityUtils.toString(ex.getResponse().getEntity()),
            containsString("Invalid space type. Please refer to the available space types")
        );
    }

    public void testMissingParamsInScript() throws Exception {
        /*
         * Create knn index and populate data
         */
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));

        /**
         * Construct Search Request
         */
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        float[] queryVector = { 2.0f, -2.0f };
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.COSINESIMIL.getValue());
        Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString("Missing parameter [field]"));

        // Remove query vector parameter
        params.put("field", FIELD_NAME);
        params.remove("query_value");
        Request vector_request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        ex = expectThrows(ResponseException.class, () -> client().performRequest(vector_request));
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString("Missing parameter [query_value]"));

        // Remove space parameter
        params.put("query_value", queryVector);
        params.remove("space_type");
        Request space_request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        ex = expectThrows(ResponseException.class, () -> client().performRequest(space_request));
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString("Missing parameter [space_type]"));
    }

    public void testUnequalDimensions() throws Exception {
        /*
         * Create knn index and populate data
         */
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] f1 = { 1.0f, -1.0f };
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, f1);

        /**
         * Construct Search Request
         */
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        float[] queryVector = { 2.0f, -2.0f, -2.0f };  // query dimension and field dimension mismatch
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.COSINESIMIL.getValue());
        Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString("does not match"));
    }

    @SuppressWarnings("unchecked")
    public void testKNNScoreForNonVectorDocument() throws Exception {
        /*
         * Create knn index and populate data
         */
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] f1 = { 1.0f, 1.0f };
        addDocWithNumericField(INDEX_NAME, "0", "price", 10);
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, f1);
        forceMergeKnnIndex(INDEX_NAME);
        /**
         * Construct Search Request
         */
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        float[] queryVector = { 2.0f, 2.0f };  // query dimension and field dimension mismatch
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.L2.getValue());
        Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody
        ).map().get("hits")).get("hits");

        List<String> docIds = hits.stream().map(hit -> ((String) ((Map<String, Object>) hit).get("_id"))).collect(Collectors.toList());
        // assert document order
        assertEquals("1", docIds.get(0));
        assertEquals("0", docIds.get(1));

        List<Double> scores = hits.stream().map(hit -> {
            Double score = ((Double) ((Map<String, Object>) hit).get("_score"));
            return score;
        }).collect(Collectors.toList());
        // assert scores
        assertEquals(0.33333, scores.get(0), 0.001);
        assertEquals(Float.MIN_VALUE, scores.get(1), 0.001);
    }

    @SuppressWarnings("unchecked")
    public void testHammingScriptScore_Long() throws Exception {
        createIndex(INDEX_NAME, Settings.EMPTY);
        String longMapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "long")
            .endObject()
            .endObject()
            .endObject()
            .toString();
        putMappingRequest(INDEX_NAME, longMapping);

        addDocWithNumericField(INDEX_NAME, "0", FIELD_NAME, 8L);
        addDocWithNumericField(INDEX_NAME, "1", FIELD_NAME, 1L);
        addDocWithNumericField(INDEX_NAME, "2", FIELD_NAME, -9_223_372_036_818_523_493L);
        addDocWithNumericField(INDEX_NAME, "3", FIELD_NAME, 1_000_000_000_000_000L);

        // Add docs without the field. These docs should not appear in top 4 of results
        addDocWithNumericField(INDEX_NAME, "4", "price", 10);
        addDocWithNumericField(INDEX_NAME, "5", "price", 10);
        addDocWithNumericField(INDEX_NAME, "6", "price", 10);

        /*
         * Decimal to Binary conversions lookup
         * 8                          -> 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1000
         * 1                          -> 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0001
         * -9_223_372_036_818_523_493 -> 1000 0000 0000 0000 0000 0000 0000 0000 0000 0010 0010 1001 0010 1010 1001 1011
         * 1_000_000_000_000_000      -> 0000 0000 0000 0011 1000 1101 0111 1110 1010 0100 1100 0110 1000 0000 0000 0000
         * -9_223_372_036_818_526_181 -> 1000 0000 0000 0000 0000 0000 0000 0000 0000 0010 0010 1001 0010 0000 0001 1011
         * 10                         -> 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1010
         */

        QueryBuilder qb1 = new MatchAllQueryBuilder();
        Map<String, Object> params1 = new HashMap<>();
        Long queryValue1 = -9223372036818526181L;
        params1.put("field", FIELD_NAME);
        params1.put("query_value", queryValue1);
        params1.put("space_type", KNNScoringSpaceFactory.HAMMING_BIT);
        Request request1 = constructKNNScriptQueryRequest(INDEX_NAME, qb1, params1, 4, Collections.emptyMap());
        Response response1 = client().performRequest(request1);
        assertEquals(request1.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response1.getStatusLine().getStatusCode()));

        String responseBody1 = EntityUtils.toString(response1.getEntity());
        List<Object> hits1 = (List<Object>) ((Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody1
        ).map().get("hits")).get("hits");

        List<String> docIds1 = hits1.stream().map(hit -> ((String) ((Map<String, Object>) hit).get("_id"))).collect(Collectors.toList());

        List<Double> docScores1 = hits1.stream()
            .map(hit -> ((Double) ((Map<String, Object>) hit).get("_score")))
            .collect(Collectors.toList());

        double[] scores1 = new double[docScores1.size()];
        for (int i = 0; i < docScores1.size(); i++) {
            scores1[i] = docScores1.get(i);
        }

        List<String> correctIds1 = Arrays.asList("2", "0", "1", "3");
        double[] correctScores1 = new double[] { 1.0 / (1 + 3), 1.0 / (1 + 9), 1.0 / (1 + 9), 1.0 / (1 + 30) };

        assertEquals(4, correctIds1.size());
        assertArrayEquals(correctIds1.toArray(), docIds1.toArray());
        assertArrayEquals(correctScores1, scores1, 0.001);

        /*
         * Force merge to one segment to confirm that docs without field are not included in the results when segment
         * is mixed with docs that have the field and docs that dont.
         */
        forceMergeKnnIndex(INDEX_NAME);

        QueryBuilder qb2 = new MatchAllQueryBuilder();
        Map<String, Object> params2 = new HashMap<>();
        Long queryValue2 = 10L;
        params2.put("field", FIELD_NAME);
        params2.put("query_value", queryValue2);
        params2.put("space_type", KNNScoringSpaceFactory.HAMMING_BIT);
        Request request2 = constructKNNScriptQueryRequest(INDEX_NAME, qb2, params2, 4, Collections.emptyMap());
        Response response2 = client().performRequest(request2);
        assertEquals(request2.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response2.getStatusLine().getStatusCode()));

        String responseBody2 = EntityUtils.toString(response2.getEntity());
        List<Object> hits2 = (List<Object>) ((Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody2
        ).map().get("hits")).get("hits");

        List<String> docIds2 = hits2.stream().map(hit -> ((String) ((Map<String, Object>) hit).get("_id"))).collect(Collectors.toList());

        List<Double> docScores2 = hits2.stream()
            .map(hit -> ((Double) ((Map<String, Object>) hit).get("_score")))
            .collect(Collectors.toList());

        double[] scores2 = new double[docScores2.size()];
        for (int i = 0; i < docScores2.size(); i++) {
            scores2[i] = docScores2.get(i);
        }

        List<String> correctIds2 = Arrays.asList("0", "1", "2", "3");
        double[] correctScores2 = new double[] { 1.0 / (1 + 1), 1.0 / (1 + 3), 1.0 / (1 + 11), 1.0 / (1 + 22) };

        assertEquals(4, correctIds2.size());
        assertArrayEquals(correctIds2.toArray(), docIds2.toArray());
        assertArrayEquals(correctScores2, scores2, 0.001);
    }

    @SuppressWarnings("unchecked")
    public void testHammingScriptScore_Base64() throws Exception {
        createIndex(INDEX_NAME, Settings.EMPTY);
        String longMapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "binary")
            .field("doc_values", true)
            .endObject()
            .endObject()
            .endObject()
            .toString();
        putMappingRequest(INDEX_NAME, longMapping);

        addDocWithBinaryField(INDEX_NAME, "0", FIELD_NAME, "AAAAAAAAAAk=");
        addDocWithBinaryField(INDEX_NAME, "1", FIELD_NAME, "AAAAAAAAAAE=");
        addDocWithBinaryField(INDEX_NAME, "2", FIELD_NAME, "gAAAAAIpKps=");
        addDocWithBinaryField(INDEX_NAME, "3", FIELD_NAME, "AAONfqTGgAA=");

        // Add docs without the field. These docs should not appear in top 4 of results
        addDocWithNumericField(INDEX_NAME, "4", "price", 10);
        addDocWithNumericField(INDEX_NAME, "5", "price", 10);
        addDocWithNumericField(INDEX_NAME, "6", "price", 10);

        /*
         * Base64 encodings to Binary conversions lookup
         * AAAAAAAAAAk=  -> 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1001
         * AAAAAAAAAAE=  -> 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0001
         * gAAAAAIpKps=  -> 1000 0000 0000 0000 0000 0000 0000 0000 0000 0010 0010 1001 0010 1010 1001 1011
         * AAONfqTGgAA=  -> 0000 0000 0000 0011 1000 1101 0111 1110 1010 0100 1100 0110 1000 0000 0000 0000
         * gAAAAAIpIBs=  -> 1000 0000 0000 0000 0000 0000 0000 0000 0000 0010 0010 1001 0010 0000 0001 1011
         * AAAAAAIpIBs=  -> 0000 0000 0000 0000 0000 0000 0000 0000 0000 0010 0010 1001 0010 0000 0001 1011
         */

        QueryBuilder qb1 = new MatchAllQueryBuilder();
        Map<String, Object> params1 = new HashMap<>();
        String queryValue1 = "gAAAAAIpIBs=";
        params1.put("field", FIELD_NAME);
        params1.put("query_value", queryValue1);
        params1.put("space_type", KNNScoringSpaceFactory.HAMMING_BIT);
        Request request1 = constructKNNScriptQueryRequest(INDEX_NAME, qb1, params1, 4, Collections.emptyMap());
        Response response1 = client().performRequest(request1);
        assertEquals(request1.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response1.getStatusLine().getStatusCode()));

        String responseBody1 = EntityUtils.toString(response1.getEntity());
        List<Object> hits1 = (List<Object>) ((Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody1
        ).map().get("hits")).get("hits");

        List<String> docIds1 = hits1.stream().map(hit -> ((String) ((Map<String, Object>) hit).get("_id"))).collect(Collectors.toList());

        List<Double> docScores1 = hits1.stream()
            .map(hit -> ((Double) ((Map<String, Object>) hit).get("_score")))
            .collect(Collectors.toList());

        double[] scores1 = new double[docScores1.size()];
        for (int i = 0; i < docScores1.size(); i++) {
            scores1[i] = docScores1.get(i);
        }

        List<String> correctIds1 = Arrays.asList("2", "0", "1", "3");
        double[] correctScores1 = new double[] { 1.0 / (1 + 3), 1.0 / (1 + 8), 1.0 / (1 + 9), 1.0 / (1 + 30) };

        assertEquals(correctIds1.size(), docIds1.size());
        assertArrayEquals(correctIds1.toArray(), docIds1.toArray());
        assertArrayEquals(correctScores1, scores1, 0.001);

        /*
         * Force merge to one segment to confirm that docs without field are not included in the results when segment
         * is mixed with docs that have the field and docs that dont.
         */
        forceMergeKnnIndex(INDEX_NAME);

        QueryBuilder qb2 = new MatchAllQueryBuilder();
        Map<String, Object> params2 = new HashMap<>();
        String queryValue2 = "AAAAAAIpIBs=";
        params2.put("field", FIELD_NAME);
        params2.put("query_value", queryValue2);
        params2.put("space_type", KNNScoringSpaceFactory.HAMMING_BIT);
        Request request2 = constructKNNScriptQueryRequest(INDEX_NAME, qb2, params2, 4, Collections.emptyMap());
        Response response2 = client().performRequest(request2);
        assertEquals(request2.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response2.getStatusLine().getStatusCode()));

        String responseBody2 = EntityUtils.toString(response2.getEntity());
        List<Object> hits2 = (List<Object>) ((Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody2
        ).map().get("hits")).get("hits");

        List<String> docIds2 = hits2.stream().map(hit -> ((String) ((Map<String, Object>) hit).get("_id"))).collect(Collectors.toList());

        List<Double> docScores2 = hits2.stream()
            .map(hit -> ((Double) ((Map<String, Object>) hit).get("_score")))
            .collect(Collectors.toList());

        double[] scores2 = new double[docScores2.size()];
        for (int i = 0; i < docScores2.size(); i++) {
            scores2[i] = docScores2.get(i);
        }

        List<String> correctIds2 = Arrays.asList("2", "0", "1", "3");
        double[] correctScores2 = new double[] { 1.0 / (1 + 4), 1.0 / (1 + 7), 1.0 / (1 + 8), 1.0 / (1 + 29) };

        assertEquals(correctIds2.size(), docIds2.size());
        assertArrayEquals(correctIds2.toArray(), docIds2.toArray());
        assertArrayEquals(correctScores2, scores2, 0.001);
    }

    public void testKNNInnerProdScriptScore() throws Exception {
        testKNNScriptScore(SpaceType.INNER_PRODUCT);
    }

    public void testKNNScriptScoreWithRequestCacheEnabled() throws Exception {
        /*
         * Create knn index and populate data
         */
        createKnnIndex(INDEX_NAME, createKnnIndexMapping(FIELD_NAME, 2));
        Float[] f1 = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, f1);

        Float[] f2 = { 2.0f, 2.0f };
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, f2);

        Float[] f3 = { 4.0f, 4.0f };
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, f3);

        Float[] f4 = { 3.0f, 3.0f };
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, f4);

        /**
         * Construct Search Request
         */
        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> scriptParams = new HashMap<>();
        /*
         *   params": {
         *       "field": "my_dense_vector",
         *       "vector": [2.0, 2.0]
         *      }
         */
        float[] queryVector = { 1.0f, 1.0f };
        scriptParams.put("field", FIELD_NAME);
        scriptParams.put("query_value", queryVector);
        scriptParams.put("space_type", SpaceType.L2.getValue());
        Map<String, Object> searchParams = new HashMap<>();
        searchParams.put("request_cache", true);

        // first request with request cache enabled
        Request firstScriptQueryRequest = constructKNNScriptQueryRequest(INDEX_NAME, qb, scriptParams, 4, searchParams);
        Response firstScriptQueryResponse = client().performRequest(firstScriptQueryRequest);
        assertEquals(
            firstScriptQueryRequest.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(firstScriptQueryResponse.getStatusLine().getStatusCode())
        );

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(firstScriptQueryResponse.getEntity()), FIELD_NAME);
        List<String> expectedDocids = Arrays.asList("2", "4", "3", "1");

        List<String> actualDocids = new ArrayList<>();
        for (KNNResult result : results) {
            actualDocids.add(result.getDocId());
        }

        assertEquals(4, results.size());
        assertEquals(expectedDocids, actualDocids);

        // assert that the request cache was hit missed at first request
        Request firstStatsRequest = new Request("GET", "/" + INDEX_NAME + "/_stats");
        Response firstStatsResponse = client().performRequest(firstStatsRequest);
        assertEquals(
            firstStatsRequest.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(firstStatsResponse.getStatusLine().getStatusCode())
        );
        String firstStatsResponseBody = EntityUtils.toString(firstStatsResponse.getEntity());
        Map<String, Object> firstQueryCacheMap = Optional.ofNullable(
            createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), firstStatsResponseBody).map()
        )
            .map(r -> (Map<String, Object>) r.get("indices"))
            .map(i -> (Map<String, Object>) i.get(INDEX_NAME))
            .map(ind -> (Map<String, Object>) ind.get("total"))
            .map(t -> (Map<String, Object>) t.get("request_cache"))
            .orElseThrow(() -> new IllegalStateException("Query Cache Map not found"));
        // assert that the request cache was hit missed at first request
        assertEquals(1, firstQueryCacheMap.get("miss_count"));
        assertEquals(0, firstQueryCacheMap.get("hit_count"));

        // second request with request cache enabled
        Request secondScriptQueryRequest = constructKNNScriptQueryRequest(INDEX_NAME, qb, scriptParams, 4, searchParams);
        Response secondScriptQueryResponse = client().performRequest(secondScriptQueryRequest);
        assertEquals(
            firstScriptQueryRequest.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(secondScriptQueryResponse.getStatusLine().getStatusCode())
        );

        Request secondStatsRequest = new Request("GET", "/" + INDEX_NAME + "/_stats");
        Response secondStatsResponse = client().performRequest(secondStatsRequest);
        assertEquals(
            secondStatsRequest.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(secondStatsResponse.getStatusLine().getStatusCode())
        );
        String secondStatsResponseBody = EntityUtils.toString(secondStatsResponse.getEntity());
        Map<String, Object> secondQueryCacheMap = Optional.ofNullable(
            createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), secondStatsResponseBody).map()
        )
            .map(r -> (Map<String, Object>) r.get("indices"))
            .map(i -> (Map<String, Object>) i.get(INDEX_NAME))
            .map(ind -> (Map<String, Object>) ind.get("total"))
            .map(t -> (Map<String, Object>) t.get("request_cache"))
            .orElseThrow(() -> new IllegalStateException("Query Cache Map not found"));
        assertEquals(1, secondQueryCacheMap.get("miss_count"));
        // assert that the request cache was hit at second request
        assertEquals(1, secondQueryCacheMap.get("hit_count"));
    }

    public void testKNNScriptScoreOnModelBasedIndex() throws Exception {
        int dimensions = randomIntBetween(2, 10);
        String trainMapping = createKnnIndexMapping(TRAIN_FIELD_PARAMETER, dimensions);
        createKnnIndex(TRAIN_INDEX_PARAMETER, trainMapping);
        bulkIngestRandomVectors(TRAIN_INDEX_PARAMETER, TRAIN_FIELD_PARAMETER, dimensions * 3, dimensions);

        XContentBuilder methodBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 4)
            .field(METHOD_PARAMETER_NPROBES, 2)
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(methodBuilder);

        trainModel(TEST_MODEL, TRAIN_INDEX_PARAMETER, TRAIN_FIELD_PARAMETER, dimensions, method, "test model for script score");
        assertTrainingSucceeds(TEST_MODEL, 30, 1000);

        String testMapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(MODEL_ID, TEST_MODEL)
            .endObject()
            .endObject()
            .endObject()
            .toString();

        for (SpaceType spaceType : SpaceType.values()) {
            if (SpaceType.UNDEFINED == spaceType || SpaceType.HAMMING == spaceType) {
                continue;
            }
            final float[] queryVector = randomVector(dimensions);
            final BiFunction<float[], float[], Float> scoreFunction = getScoreFunction(spaceType, queryVector);
            createIndexAndAssertScriptScore(testMapping, spaceType, scoreFunction, dimensions, queryVector, true);
        }
    }

    private List<String> createMappers(int dimensions) throws Exception {
        return List.of(
            createKnnIndexMapping(FIELD_NAME, dimensions),
            createKnnIndexMapping(
                FIELD_NAME,
                dimensions,
                KNNConstants.METHOD_HNSW,
                KNNEngine.LUCENE.getName(),
                SpaceType.DEFAULT.getValue(),
                true
            ),
            createKnnIndexMapping(
                FIELD_NAME,
                dimensions,
                KNNConstants.METHOD_HNSW,
                KNNEngine.LUCENE.getName(),
                SpaceType.DEFAULT.getValue(),
                false
            )
        );
    }

    private List<String> createBinaryIndexMappers(int dimensions) throws Exception {
        return List.of(
            createKnnIndexMapping(
                FIELD_NAME,
                dimensions,
                KNNConstants.METHOD_HNSW,
                KNNEngine.FAISS.getName(),
                SpaceType.DEFAULT_BINARY.getValue(),
                true,
                VectorDataType.BINARY
            ),
            createKnnIndexMapping(
                FIELD_NAME,
                dimensions,
                KNNConstants.METHOD_HNSW,
                KNNEngine.FAISS.getName(),
                SpaceType.DEFAULT_BINARY.getValue(),
                false,
                VectorDataType.BINARY
            )
        );
    }

    private float[] randomVector(final int dimensions) {
        return randomVector(dimensions, VectorDataType.FLOAT);
    }

    private float[] randomVector(final int dimensions, final VectorDataType vectorDataType) {
        int size = VectorDataType.BINARY == vectorDataType ? dimensions / 8 : dimensions;
        final float[] vector = new float[size];
        for (int i = 0; i < size; i++) {
            vector[i] = VectorDataType.FLOAT == vectorDataType ? randomFloat() : randomByte();
        }
        return vector;
    }

    private Map<String, KNNResult> createDataset(
        Function<float[], Float> scoreFunction,
        int dimensions,
        int numDocsWithField,
        boolean dense,
        VectorDataType vectorDataType
    ) {
        final Map<String, KNNResult> dataset = new HashMap<>(dense ? numDocsWithField : numDocsWithField * 3);
        int id = 0;
        for (int i = 0; i < numDocsWithField; i++) {
            final int dummyDocs = dense ? 0 : randomIntBetween(2, 5);
            for (int j = 0; j < dummyDocs; j++) {
                dataset.put(Integer.toString(id++), null);
            }
            final float[] vector = randomVector(dimensions, vectorDataType);
            final float score = scoreFunction.apply(vector);
            dataset.put(Integer.toString(id), new KNNResult(Integer.toString(id++), vector, score));
        }
        return dataset;
    }

    private BiFunction<float[], float[], Float> getScoreFunction(SpaceType spaceType, float[] queryVector) {
        KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            SpaceType.HAMMING == spaceType ? queryVector.length * 8 : queryVector.length,
            SpaceType.HAMMING == spaceType ? VectorDataType.BINARY : VectorDataType.FLOAT,
            null
        );
        List<Float> target = new ArrayList<>(queryVector.length);
        for (float f : queryVector) {
            target.add(f);
        }
        KNNScoringSpace knnScoringSpace = KNNScoringSpaceFactory.create(spaceType.getValue(), target, knnVectorFieldType);
        switch (spaceType) {
            case L1:
            case L2:
            case LINF:
            case COSINESIMIL:
            case INNER_PRODUCT:
            case HAMMING:
                return ((KNNScoringSpace.KNNFieldSpace) knnScoringSpace).getScoringMethod();
            default:
                throw new IllegalArgumentException();
        }
    }

    private void testKNNScriptScore(SpaceType spaceType) throws Exception {
        final int dims = randomIntBetween(2, 10);
        final float[] queryVector = randomVector(dims);
        final BiFunction<float[], float[], Float> scoreFunction = getScoreFunction(spaceType, queryVector);
        for (String mapper : createMappers(dims)) {
            createIndexAndAssertScriptScore(mapper, spaceType, scoreFunction, dims, queryVector, true);
            createIndexAndAssertScriptScore(mapper, spaceType, scoreFunction, dims, queryVector, false);
        }
    }

    private void testKNNScriptScoreOnBinaryIndex(SpaceType spaceType) throws Exception {
        final int dims = randomIntBetween(2, 10) * 8;
        final float[] queryVector = randomVector(dims, VectorDataType.BINARY);
        final BiFunction<float[], float[], Float> scoreFunction = getScoreFunction(spaceType, queryVector);

        // Test when knn is enabled and engine is Faiss
        for (String mapper : createBinaryIndexMappers(dims)) {
            createIndexAndAssertScriptScore(mapper, spaceType, scoreFunction, dims, queryVector, true, true, VectorDataType.BINARY);
            createIndexAndAssertScriptScore(mapper, spaceType, scoreFunction, dims, queryVector, false, true, VectorDataType.BINARY);
        }

        // Test when knn is disabled and engine is default(Nmslib)
        createIndexAndAssertScriptScore(
            createKnnIndexMapping(FIELD_NAME, dims, VectorDataType.BINARY),
            spaceType,
            scoreFunction,
            dims,
            queryVector,
            true,
            false,
            VectorDataType.BINARY
        );
        createIndexAndAssertScriptScore(
            createKnnIndexMapping(FIELD_NAME, dims, VectorDataType.BINARY),
            spaceType,
            scoreFunction,
            dims,
            queryVector,
            false,
            false,
            VectorDataType.BINARY
        );
    }

    private void createIndexAndAssertScriptScore(
        String mapper,
        SpaceType spaceType,
        BiFunction<float[], float[], Float> scoreFunction,
        int dimensions,
        float[] queryVector,
        boolean dense
    ) throws Exception {
        createIndexAndAssertScriptScore(mapper, spaceType, scoreFunction, dimensions, queryVector, dense, true, VectorDataType.FLOAT);
    }

    private void createIndexAndAssertScriptScore(
        String mapper,
        SpaceType spaceType,
        BiFunction<float[], float[], Float> scoreFunction,
        int dimensions,
        float[] queryVector,
        boolean dense,
        boolean enableKnn,
        VectorDataType vectorDataType
    ) throws Exception {
        /*
         * Create knn index and populate data
         */
        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", enableKnn).build();
        createKnnIndex(INDEX_NAME, settings, mapper);
        try {
            final int numDocsWithField = randomIntBetween(4, 10);
            Map<String, KNNResult> dataset = createDataset(
                v -> scoreFunction.apply(queryVector, v),
                dimensions,
                numDocsWithField,
                dense,
                vectorDataType
            );
            final float[] dummyVector = new float[1];
            dataset.forEach((k, v) -> {
                final float[] vector = (v != null) ? v.getVector() : dummyVector;
                ExceptionsHelper.catchAsRuntimeException(() -> addKnnDoc(INDEX_NAME, k, (v != null) ? FIELD_NAME : "dummy", vector));
            });

            /**
             * Construct Search Request
             */
            QueryBuilder qb = new MatchAllQueryBuilder();
            Map<String, Object> params = new HashMap<>();
            /*
             *   params": {
             *       "field": FIELD_NAME,
             *       "vector": queryVector
             *      }
             */
            params.put("field", FIELD_NAME);
            params.put("query_value", queryVector);
            params.put("space_type", spaceType.getValue());
            Request request = constructKNNScriptQueryRequest(INDEX_NAME, qb, params, numDocsWithField);
            Response response = client().performRequest(request);
            assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
            assertTrue(results.stream().allMatch(r -> dataset.get(r.getDocId()).equals(r)));
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }
}
