/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.script.Script;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;

public class VectorDataTypeIT extends KNNRestTestCase {
    private static final String INDEX_NAME = "test-index-vec-dt";
    private static final String FIELD_NAME = "test-field-vec-dt";
    private static final String PROPERTIES_FIELD = "properties";
    private static final String DOC_ID = "doc1";
    private static final String TYPE_FIELD_NAME = "type";
    private static final String KNN_VECTOR_TYPE = "knn_vector";
    private static final int EF_CONSTRUCTION = 128;
    private static final int M = 16;
    private static final QueryBuilder MATCH_ALL_QUERY_BUILDER = new MatchAllQueryBuilder();

    @After
    @SneakyThrows
    public final void cleanUp() {
        deleteKNNIndex(INDEX_NAME);
    }

    // Validate if we are able to create an index by setting data_type field as byte and add a doc to it
    @SneakyThrows
    public void testAddDocWithByteVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Byte[] vector = { 6, 6 };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        refreshAllIndices();
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    // Validate by creating an index by setting data_type field as byte, add a doc to it and update it later.
    @SneakyThrows
    public void testUpdateDocWithByteVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Byte[] vector = { -36, 78 };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        Byte[] updatedVector = { 89, -8 };
        updateKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, updatedVector);

        refreshAllIndices();
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    // Validate by creating an index by setting data_type field as byte, add a doc to it and delete it later.
    @SneakyThrows
    public void testDeleteDocWithByteVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Byte[] vector = { 35, -46 };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        deleteKnnDoc(INDEX_NAME, DOC_ID);
        refreshAllIndices();

        assertEquals(0, getDocCount(INDEX_NAME));
    }

    @SneakyThrows
    public void testSearchWithByteVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        ingestL2ByteTestData();

        Byte[] queryVector = { 1, 1 };
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, convertByteToFloatArray(queryVector), 4), 4);

        validateL2SearchResults(response);
    }

    @SneakyThrows
    public void testSearchWithInvalidByteVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        ingestL2ByteTestData();

        // Validate search with floats instead of byte vectors
        float[] queryVector = { -10.76f, 15.89f };
        ResponseException ex = expectThrows(
            ResponseException.class,
            () -> searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 4), 4)
        );
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue()
                    )
                )
        );

        // validate search with search vectors outside of byte range
        float[] queryVector1 = { -1000.0f, 200.0f };
        ResponseException ex1 = expectThrows(
            ResponseException.class,
            () -> searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector1, 4), 4)
        );

        assertTrue(
            ex1.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [%d, %d]",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue(),
                        Byte.MIN_VALUE,
                        Byte.MAX_VALUE
                    )
                )
        );
    }

    @SneakyThrows
    public void testSearchWithFloatVectorDataType() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.FLOAT.getValue());
        ingestL2FloatTestData();

        float[] queryVector = { 1.0f, 1.0f };
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 4), 4);

        validateL2SearchResults(response);
    }

    // Set an invalid value for data_type field while creating the index which should throw an exception
    public void testInvalidVectorDataType() {
        String vectorDataType = "invalidVectorType";
        ResponseException ex = expectThrows(
            ResponseException.class,
            () -> createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, vectorDataType)
        );
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "Invalid value provided for [%s] field. Supported values are [%s]",
                        VECTOR_DATA_TYPE_FIELD,
                        SUPPORTED_VECTOR_DATA_TYPES
                    )
                )
        );
    }

    // Set null value for data_type field while creating the index which should throw an exception
    public void testVectorDataTypeAsNull() {
        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, null));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] on mapper [%s] of type [%s] must not have a [null] value",
                        VECTOR_DATA_TYPE_FIELD,
                        FIELD_NAME,
                        KNN_VECTOR_TYPE
                    )
                )
        );
    }

    // Create an index with byte vector data_type and add a doc with decimal values which should throw exception
    @SneakyThrows
    public void testInvalidVectorData() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Float[] vector = { -10.76f, 15.89f };

        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue()
                    )
                )
        );
    }

    // Create an index with byte vector data_type and add a doc with values out of byte range which should throw exception
    @SneakyThrows
    public void testInvalidByteVectorRange() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        Float[] vector = { -1000f, 155f };

        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [%d, %d]",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue(),
                        Byte.MIN_VALUE,
                        Byte.MAX_VALUE
                    )
                )
        );
    }

    // Create an index with byte vector data_type using nmslib engine which should throw an exception
    public void testByteVectorDataTypeWithNmslibEngine() {
        ResponseException ex = expectThrows(
            ResponseException.class,
            () -> createKnnIndexMappingWithNmslibEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue())
        );
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field with value [%s] is only supported for [%s] engine",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue(),
                        LUCENE_NAME
                    )
                )
        );
    }

    @SneakyThrows
    public void testByteVectorDataTypeWithLegacyFieldMapperKnnIndexSetting() {
        // Create an index with byte vector data_type and index.knn as true without setting KnnMethodContext,
        // which should throw an exception because the LegacyFieldMapper will use NMSLIB engine and byte data_type
        // is not supported for NMSLIB engine.
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, 2)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.BYTE.getValue())
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();

        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field with value [%s] is not supported for [%s] engine",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue(),
                        NMSLIB_NAME
                    )
                )
        );

    }

    public void testDocValuesWithByteVectorDataTypeLuceneEngine() throws Exception {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.BYTE.getValue());
        ingestL2ByteTestData();

        Byte[] queryVector = { 1, 1 };
        Request request = createScriptQueryRequest(queryVector, SpaceType.L2.getValue(), MATCH_ALL_QUERY_BUILDER);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        validateL2SearchResults(response);
    }

    public void testDocValuesWithFloatVectorDataTypeLuceneEngine() throws Exception {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.FLOAT.getValue());
        ingestL2FloatTestData();

        Byte[] queryVector = { 1, 1 };
        Request request = createScriptQueryRequest(queryVector, SpaceType.L2.getValue(), MATCH_ALL_QUERY_BUILDER);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        validateL2SearchResults(response);
    }

    public void testL2ScriptScoreWithByteVectorDataType() throws Exception {
        createKnnIndexMappingForScripting(2, VectorDataType.BYTE.getValue());
        ingestL2ByteTestData();

        Byte[] queryVector = { 1, 1 };
        Request request = createScriptQueryRequest(queryVector, SpaceType.L2.getValue(), MATCH_ALL_QUERY_BUILDER);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        validateL2SearchResults(response);
    }

    public void testL2ScriptScoreWithFloatVectorDataType() throws Exception {
        createKnnIndexMappingForScripting(2, VectorDataType.FLOAT.getValue());
        ingestL2FloatTestData();

        Float[] queryVector = { 1.0f, 1.0f };
        Request request = createScriptQueryRequest(queryVector, SpaceType.L2.getValue(), MATCH_ALL_QUERY_BUILDER);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        validateL2SearchResults(response);
    }

    public void testL2PainlessScriptingWithByteVectorDataType() throws Exception {
        createKnnIndexMappingForScripting(2, VectorDataType.BYTE.getValue());
        ingestL2ByteTestData();

        String source = String.format("1/(1 + l2Squared([1, 1], doc['%s']))", FIELD_NAME);
        Request request = constructScriptScoreContextSearchRequest(
            INDEX_NAME,
            MATCH_ALL_QUERY_BUILDER,
            Collections.emptyMap(),
            Script.DEFAULT_SCRIPT_LANG,
            source,
            4,
            Collections.emptyMap()
        );

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        validateL2SearchResults(response);
    }

    public void testL2PainlessScriptingWithFloatVectorDataType() throws Exception {
        createKnnIndexMappingForScripting(2, VectorDataType.FLOAT.getValue());
        ingestL2FloatTestData();

        String source = String.format("1/(1 + l2Squared([1.0f, 1.0f], doc['%s']))", FIELD_NAME);
        Request request = constructScriptScoreContextSearchRequest(
            INDEX_NAME,
            MATCH_ALL_QUERY_BUILDER,
            Collections.emptyMap(),
            Script.DEFAULT_SCRIPT_LANG,
            source,
            4,
            Collections.emptyMap()
        );

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        validateL2SearchResults(response);
    }

    public void testKNNScriptScoreWithInvalidVectorDataType() {
        // Set an invalid value for data_type field while creating the index for script scoring which should throw an exception
        ResponseException ex = expectThrows(ResponseException.class, () -> createKnnIndexMappingForScripting(2, "invalid_data_type"));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "Invalid value provided for [%s] field. Supported values are [%s]",
                        VECTOR_DATA_TYPE_FIELD,
                        SUPPORTED_VECTOR_DATA_TYPES
                    )
                )
        );
    }

    public void testKNNScriptScoreWithInvalidByteQueryVector() throws Exception {
        // Create an index with byte vector data_type, add docs and run a scoring script query with decimal values
        // which should throw exception
        createKnnIndexMappingForScripting(2, VectorDataType.BYTE.getValue());

        Byte[] f1 = { 6, 6 };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, f1);

        Byte[] f2 = { 2, 2 };
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, f2);

        // Construct Search Request with query vector having decimal values
        Float[] queryVector = { 10.67f, 19.78f };
        Request request = createScriptQueryRequest(queryVector, SpaceType.L2.getValue(), MATCH_ALL_QUERY_BUILDER);
        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue()
                    )
                )
        );
    }

    @SneakyThrows
    public void testSearchWithInvalidSearchVectorType() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.FLOAT.getValue());
        ingestL2FloatTestData();
        Request request = new Request("POST", String.format("/%s/_search", INDEX_NAME));
        List<Object> invalidTypeQueryVector = new ArrayList<>();
        invalidTypeQueryVector.add(1.5);
        invalidTypeQueryVector.add(2.5);
        invalidTypeQueryVector.add("a");
        invalidTypeQueryVector.add(null);
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", invalidTypeQueryVector)
            .field("k", 4)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        request.setJsonEntity(builder.toString());

        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertEquals(400, ex.getResponse().getStatusLine().getStatusCode());
        assertTrue(ex.getMessage(), ex.getMessage().contains("[knn] failed to parse field [vector]"));
    }

    @SneakyThrows
    public void testSearchWithMissingQueryVector() {
        createKnnIndexMappingWithLuceneEngine(2, SpaceType.L2, VectorDataType.FLOAT.getValue());
        ingestL2FloatTestData();
        Request request = new Request("POST", String.format("/%s/_search", INDEX_NAME));
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("k", 4)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        request.setJsonEntity(builder.toString());

        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertEquals(400, ex.getResponse().getStatusLine().getStatusCode());
        assertTrue(ex.getMessage().contains("[knn] requires query vector"));
    }

    @SneakyThrows
    private void ingestL2ByteTestData() {
        Byte[] b1 = { 6, 6 };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, b1);

        Byte[] b2 = { 2, 2 };
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, b2);

        Byte[] b3 = { 4, 4 };
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, b3);

        Byte[] b4 = { 3, 3 };
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, b4);
    }

    @SneakyThrows
    private void ingestL2FloatTestData() {
        Float[] f1 = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, f1);

        Float[] f2 = { 2.0f, 2.0f };
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, f2);

        Float[] f3 = { 4.0f, 4.0f };
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, f3);

        Float[] f4 = { 3.0f, 3.0f };
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, f4);
    }

    private void createKnnIndexMappingWithNmslibEngine(int dimension, SpaceType spaceType, String vectorDataType) throws Exception {
        createKnnIndexMappingWithCustomEngine(dimension, spaceType, vectorDataType, KNNEngine.NMSLIB.getName());
    }

    private void createKnnIndexMappingWithLuceneEngine(int dimension, SpaceType spaceType, String vectorDataType) throws Exception {
        createKnnIndexMappingWithCustomEngine(dimension, spaceType, vectorDataType, KNNEngine.LUCENE.getName());
    }

    private void createKnnIndexMappingWithCustomEngine(int dimension, SpaceType spaceType, String vectorDataType, String engine)
        throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, vectorDataType)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, engine)
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createKnnIndexMappingForScripting(int dimension, String vectorDataType) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, vectorDataType)
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, Settings.EMPTY, mapping);
    }

    @SneakyThrows
    private Request createScriptQueryRequest(Byte[] queryVector, String spaceType, QueryBuilder qb) {
        Map<String, Object> params = new HashMap<>();
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", spaceType);
        return constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
    }

    @SneakyThrows
    private Request createScriptQueryRequest(Float[] queryVector, String spaceType, QueryBuilder qb) {
        Map<String, Object> params = new HashMap<>();
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", spaceType);
        return constructKNNScriptQueryRequest(INDEX_NAME, qb, params);
    }

    @SneakyThrows
    private void validateL2SearchResults(Response response) {

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertEquals(4, results.size());

        String[] expectedDocIDs = { "2", "4", "3", "1" };
        for (int i = 0; i < results.size(); i++) {
            assertEquals(expectedDocIDs[i], results.get(i).getDocId());
        }
    }

    private float[] convertByteToFloatArray(Byte[] arr) {
        float[] floatArray = new float[arr.length];
        for (int i = 0; i < arr.length; i++) {
            floatArray[i] = arr[i];
        }
        return floatArray;
    }
}
