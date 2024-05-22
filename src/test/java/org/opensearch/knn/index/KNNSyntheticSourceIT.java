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

package org.opensearch.knn.index;

import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;

import static org.opensearch.knn.common.Constants.FIELD_FILTER;
import static org.opensearch.knn.common.Constants.FIELD_TERM;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNConstants.PATH;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE_NESTED;
import static org.opensearch.knn.common.KNNConstants.VECTOR;

public class KNNSyntheticSourceIT extends KNNRestTestCase {

    static final String fieldName = "test-field-1";
    static final String nestedPath = "nested-field";
    static final String nestedFieldName = "test-nested-field-1";
    static final String nestedField = nestedPath + "." + nestedFieldName;

    public void testSyntheticSourceSearch_whenEnabledSynthetic_thenReturnSource() throws IOException, ParseException {
        String indexNameWithSynthetic = "test-index-synthetic";

        // Create an index
        XContentBuilder builder = constructMappingBuilder();

        String mapping = builder.toString();
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", true)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithSynthetic, indexSettingWithSynthetic, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(indexNameWithSynthetic, "1", fieldName, vector);
        float[] queryVector = { 6.0f, 6.0f };

        Response responseWithSynthetic = searchKNNIndex(indexNameWithSynthetic, new KNNQueryBuilder(fieldName, queryVector, 10), 10);
        String resp1 = EntityUtils.toString(responseWithSynthetic.getEntity());
        assertTrue(resp1.contains("\"test-field-1\":[6.0,6.0]"));
    }

    public void testSyntheticSourceSearch_whenDisabledSynthetic_thenReturnNoSource() throws IOException, ParseException {
        String indexNameWithoutSynthetic = "test-index-no-synthetic";

        // Create an index
        XContentBuilder builder = constructMappingBuilder();

        String mapping = builder.toString();
        Settings indexSettingWithoutSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", false)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithoutSynthetic, indexSettingWithoutSynthetic, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(indexNameWithoutSynthetic, "1", fieldName, vector);
        float[] queryVector = { 6.0f, 6.0f };

        Response responseWithoutSynthetic = searchKNNIndex(indexNameWithoutSynthetic, new KNNQueryBuilder(fieldName, queryVector, 10), 10);
        String resp2 = EntityUtils.toString(responseWithoutSynthetic.getEntity());
        assertFalse(resp2.contains("\"test-field-1\":[6.0,6.0]"));
    }

    public void testSyntheticSourceReindex_whenEnabledSynthetic_thenSuccess() throws IOException, ParseException {
        String indexNameWithSynthetic = "test-index-synthetic";
        String reindexNameWithSynthetic = "test-reindex-synthetic";

        // Create an index
        XContentBuilder builder = constructMappingBuilder();

        String mapping = builder.toString();
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", true)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithSynthetic, indexSettingWithSynthetic, mapping);
        createKnnIndex(reindexNameWithSynthetic, indexSettingWithSynthetic, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(indexNameWithSynthetic, "1", fieldName, vector);
        float[] queryVector = { 6.0f, 6.0f };

        doReindex(indexNameWithSynthetic, reindexNameWithSynthetic);

        Response responseWithSynthetic = searchKNNIndex(reindexNameWithSynthetic, new KNNQueryBuilder(fieldName, queryVector, 10), 10);
        String resp1 = EntityUtils.toString(responseWithSynthetic.getEntity());
        assertTrue(resp1.contains("\"test-field-1\":[6.0,6.0]"));
    }

    public void testSyntheticSourceReindex_whenDisableSynthetic_thenFailed() throws IOException, ParseException {
        String indexNameWithoutSynthetic = "test-index-no-synthetic";
        String reindexNameWithoutSynthetic = "test-reindex-no-synthetic";
        String fieldName = "test-field-1";

        // Create an index
        XContentBuilder builder = constructMappingBuilder();

        String mapping = builder.toString();
        Settings indexSettingWithoutSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", false)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithoutSynthetic, indexSettingWithoutSynthetic, mapping);
        createKnnIndex(reindexNameWithoutSynthetic, indexSettingWithoutSynthetic, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(indexNameWithoutSynthetic, "1", fieldName, vector);
        float[] queryVector = { 6.0f, 6.0f };

        doReindex(indexNameWithoutSynthetic, reindexNameWithoutSynthetic);

        Response responseWithoutSynthetic = searchKNNIndex(
            reindexNameWithoutSynthetic,
            new KNNQueryBuilder(fieldName, queryVector, 10),
            10
        );
        String resp2 = EntityUtils.toString(responseWithoutSynthetic.getEntity());
        assertFalse(resp2.contains("\"test-field-1\":[6.0,6.0]"));
    }

    public void testNestedFieldSyntheticSourceSearch_whenEnabledSynthetic_thenReturnSourceSuccess() throws IOException, ParseException {
        String indexNameWithSynthetic = "test-index-nested-field-synthetic";
        // Create index
        XContentBuilder builder = constructNestedMappingBuilder();

        String mapping = builder.toString();
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", true)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithSynthetic, indexSettingWithSynthetic, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDocWithTwoNestedField(indexNameWithSynthetic, "1", nestedField, vector, vector);

        Response responseWithSynthetic = queryNestedField(indexNameWithSynthetic, 10, vector);
        String resp1 = EntityUtils.toString(responseWithSynthetic.getEntity());
        assertTrue(resp1.contains("\"nested-field\":[{\"test-nested-field-1\":[6.0,6.0]},{\"test-nested-field-1\":[6.0,6.0]}]"));
    }

    public void testNestedFieldSyntheticSourceSearch_whenDisabledSynthetic_thenReturnNothingSuccess() throws IOException, ParseException {
        String indexNameWithSynthetic = "test-index-nested-field-synthetic";
        // Create index
        XContentBuilder builder = constructNestedMappingBuilder();

        String mapping = builder.toString();
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", false)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithSynthetic, indexSettingWithSynthetic, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDocWithTwoNestedField(indexNameWithSynthetic, "1", nestedField, vector, vector);

        Response responseWithSynthetic = queryNestedField(indexNameWithSynthetic, 10, vector);
        String resp1 = EntityUtils.toString(responseWithSynthetic.getEntity());
        assertFalse(resp1.contains("\"nested-field\":[{\"test-nested-field-1\":[6.0,6.0]},{\"test-nested-field-1\":[6.0,6.0]}]"));
    }

    public void testMultiNestedField_whenEnabledSynthetic_thenReturnSuccess() throws IOException, ParseException {
        String indexNameWithSynthetic = "test-index-nested-field-synthetic";
        // Create index
        XContentBuilder builder = constructNestedMappingBuilder();

        String mapping = builder.toString();
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", true)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithSynthetic, indexSettingWithSynthetic, mapping);

        /*
          "nested_field" : [
            {"nested_numeric": 1, "nested_vector": [2.6, 2.6]},
            {"nested_numeric": 2, "nested_vector": [3.1, 2.3]}
          ]
         */
        Float[] vector = { 6.0f, 6.0f };
        String[] fieldParts = nestedField.split("\\.");
        XContentBuilder docBuilder = XContentFactory.jsonBuilder().startObject();
        docBuilder.startArray(fieldParts[0]);
        docBuilder.startObject();
        docBuilder.field("nested_numeric", 1.0);
        docBuilder.field(fieldParts[1], vector);
        docBuilder.endObject();
        docBuilder.startObject();
        docBuilder.field("nested_numeric", 2.0);
        docBuilder.field(fieldParts[1], vector);
        docBuilder.endObject();
        docBuilder.endArray();
        docBuilder.endObject();

        addKnnDocWithBuilder(indexNameWithSynthetic, "1", docBuilder);

        Response responseWithSynthetic = queryNestedField(indexNameWithSynthetic, 10, vector, null, null, null, RestStatus.OK);

        String resp1 = EntityUtils.toString(responseWithSynthetic.getEntity());
        assertTrue(
            resp1.contains(
                "{\"nested-field\":[{\"nested_numeric\":1.0,\"test-nested-field-1\":[6.0,6.0]},{\"nested_numeric\":2.0,\"test-nested-field-1\":[6.0,6.0]}]}}]}"
            )
        );
    }

    public void testMultiNestedFieldWithNull_whenEnabledSynthetic_thenReturnFailed() throws IOException, ParseException {
        String indexNameWithSynthetic = "test-index-nested-field-synthetic";
        // Create index
        XContentBuilder builder = constructNestedMappingBuilder();

        String mapping = builder.toString();
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", true)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithSynthetic, indexSettingWithSynthetic, mapping);

        /*
          "nested_field" : [
            {"nested_vector": [2.6, 2.6]},
            {"nested_numeric": 2, "nested_vector": [3.1, 2.3]}
          ]
         */
        Float[] vector = { 6.0f, 6.0f };
        String[] fieldParts = nestedField.split("\\.");
        XContentBuilder docBuilder = XContentFactory.jsonBuilder().startObject();
        docBuilder.startArray(fieldParts[0]);
        docBuilder.startObject();
        docBuilder.field(fieldParts[1], vector);
        docBuilder.endObject();
        docBuilder.startObject();
        docBuilder.field("nested_numeric", 2.0);
        docBuilder.field(fieldParts[1], vector);
        docBuilder.endObject();
        docBuilder.endArray();
        docBuilder.endObject();

        addKnnDocWithBuilder(indexNameWithSynthetic, "1", docBuilder);

        try {
            Response responseWithSynthetic = queryNestedField(
                indexNameWithSynthetic,
                10,
                vector,
                null,
                null,
                null,
                RestStatus.INTERNAL_SERVER_ERROR
            );

            if (responseWithSynthetic != null) {
                // need throw exception
                assertFalse(true);
            }
        } catch (ResponseException ex) {
            assertTrue(
                ex.toString().contains("\"type\":\"unsupported_operation_exception\",\"reason\":\"Nested Field should not be empty\"")
            );
        }
    }

    public void testSyntheticSourceUpdate_whenEnabledSynthetic_thenReturnSource() throws IOException, ParseException {
        String indexNameWithSynthetic = "test-index-synthetic";
        String fieldName = "test-field-1";
        Integer dimension = 2;

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L2;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .startArray("excludes")
            .value(fieldName)
            .endArray()
            .endObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", true)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithSynthetic, indexSettingWithSynthetic, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(indexNameWithSynthetic, "1", fieldName, vector);
        float[] queryVector = { 6.0f, 6.0f };

        Response responseWithSynthetic = searchKNNIndex(indexNameWithSynthetic, new KNNQueryBuilder(fieldName, queryVector, 10), 10);
        String resp1 = EntityUtils.toString(responseWithSynthetic.getEntity());
        assertTrue(resp1.contains("\"test-field-1\":[6.0,6.0]"));

        Float[] vector2 = { 8.0f, 8.0f };
        updateKnnDoc(indexNameWithSynthetic, "1", fieldName, vector2);
        float[] queryVector2 = { 8.0f, 8.0f };
        Response responseAfterUpdate = searchKNNIndex(indexNameWithSynthetic, new KNNQueryBuilder(fieldName, queryVector2, 10), 10);
        String respUpdate = EntityUtils.toString(responseAfterUpdate.getEntity());
        assertTrue(respUpdate.contains("\"test-field-1\":[8.0,8.0]"));
    }

    public void testSyntheticSourceUpdateOtherField_whenEnabledSynthetic_thenReturnNothing() throws IOException, ParseException {
        String indexNameWithSynthetic = "test-index-synthetic";
        String fieldName = "test-field-1";
        String fieldName2 = "test-field-2";
        Integer dimension = 2;

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L2;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .startArray("excludes")
            .value(fieldName)
            .endArray()
            .endObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject(fieldName2)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        Settings indexSettingWithSynthetic = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.synthetic_source.enabled", true)
            .put("index.knn", true)
            .build();

        createKnnIndex(indexNameWithSynthetic, indexSettingWithSynthetic, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(indexNameWithSynthetic, "1", fieldName, vector);
        float[] queryVector = { 6.0f, 6.0f };

        Response responseWithSynthetic = searchKNNIndex(indexNameWithSynthetic, new KNNQueryBuilder(fieldName, queryVector, 10), 10);
        String resp1 = EntityUtils.toString(responseWithSynthetic.getEntity());
        assertTrue(resp1.contains("\"test-field-1\":[6.0,6.0]"));

        Float[] vector2 = { 8.0f, 8.0f };
        updateKnnDoc(indexNameWithSynthetic, "1", fieldName2, vector2);
        float[] queryVector2 = { 8.0f, 8.0f };
        Response responseAfterUpdate = searchKNNIndex(indexNameWithSynthetic, new KNNQueryBuilder(fieldName2, queryVector2, 10), 10);
        String respUpdate = EntityUtils.toString(responseAfterUpdate.getEntity());
        assertTrue(respUpdate.contains("\"test-field-2\":[8.0,8.0]"));
        assertFalse(respUpdate.contains("\"test-field-1\""));
    }

    private Response queryNestedField(final String index, final int k, final Object[] vector) throws IOException {
        return queryNestedField(index, k, vector, null, null, null, RestStatus.OK);
    }

    private Response queryNestedField(
        final String index,
        final Integer k,
        final Object[] vector,
        final String filterName,
        final String filterValue,
        final Float minScore,
        RestStatus Expectstatus
    ) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
        builder.startObject(TYPE_NESTED);
        builder.field(PATH, nestedPath);
        builder.startObject(QUERY).startObject(KNN).startObject(nestedPath + "." + nestedFieldName);
        builder.field(VECTOR, vector);
        if (minScore != null) {
            builder.field(MIN_SCORE, minScore);
        } else if (k != null) {
            builder.field(K, k);
        } else {
            throw new IllegalArgumentException("k or minScore must be provided in the query");
        }
        if (filterName != null && filterValue != null) {
            builder.startObject(FIELD_FILTER);
            builder.startObject(FIELD_TERM);
            builder.field(filterName, filterValue);
            builder.endObject();
            builder.endObject();
        }
        builder.endObject().endObject().endObject().endObject().endObject().endObject();
        String requestStr = builder.toString();
        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(requestStr);
        Response response;

        response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", Expectstatus, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }

    /**
     * Add a single KNN Doc to an index with two nested vector field
     *
     * @param index           name of the index
     * @param docId           id of the document
     * @param nestedFieldPath path of the nested field, e.g. "my_nested_field.my_vector"
     * @param vector1          vector to add
     * @param vector2          vector to add
     *
     */
    private void addKnnDocWithTwoNestedField(String index, String docId, String nestedFieldPath, Object[] vector1, Object[] vector2)
        throws IOException {
        String[] fieldParts = nestedFieldPath.split("\\.");

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder.startArray(fieldParts[0]);
        builder.startObject();
        builder.field(fieldParts[1], vector1);
        builder.endObject();
        builder.startObject();
        builder.field(fieldParts[1], vector2);
        builder.endObject();

        builder.endArray();
        builder.endObject();
        addKnnDocWithBuilder(index, docId, builder);
    }

    private void addKnnDocWithBuilder(String index, String docId, XContentBuilder builder) throws IOException {

        Request request = new Request("POST", "/" + index + "/_doc/" + docId + "?refresh=true");
        String docStr = builder.toString();
        request.setJsonEntity(docStr);
        client().performRequest(request);

        request = new Request("POST", "/" + index + "/_refresh");
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    private XContentBuilder constructNestedMappingBuilder() throws IOException {
        Integer dimension = 2;

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L2;
        /*
          "mappings":{
            "_source":{
              "excludes":[nestedFieldName]
            },
            "properties:{
              "nestedField":{
                "type":"nested",
                "properties":{
                  "nestedFieldName":{
                    "type":"knn_vector",
                    "dimension":2
                  }
                }
              }
            }
          }
         */
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .startArray("excludes")
            .value(nestedField)
            .endArray()
            .endObject()
            .startObject("properties")
            .startObject(nestedPath)
            .field("type", "nested")
            .startObject("properties")
            .startObject(nestedFieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return builder;
    }

    private XContentBuilder constructMappingBuilder() throws IOException {
        Integer dimension = 2;

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L2;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .startArray("excludes")
            .value(fieldName)
            .endArray()
            .endObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return builder;
    }
}
