/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.search.processor.KNNSourceExcludesProcessor;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.search.pipeline.SearchPipelineService.ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING;

/**
 * Integration tests for KNNSourceExcludesProcessor that verifies vector fields are automatically
 * excluded from _source in search responses when the system generated processor is enabled.
 */
public class KNNSourceExcludesProcessorIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "test-source-excludes";
    private static final String VECTOR_FIELD = "my_vector";
    private static final String TEXT_FIELD = "title";
    private static final int DIMENSION_VALUE = 4;
    private static final Float[] TEST_VECTOR = { 1.0f, 2.0f, 3.0f, 4.0f };

    @Before
    @SneakyThrows
    public void enableSourceExcludesProcessor() {
        updateClusterSettings(
            ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING.getKey(),
            new String[] { KNNSourceExcludesProcessor.Factory.TYPE }
        );
    }

    @After
    @SneakyThrows
    public void disableSourceExcludesProcessor() {
        updateClusterSettings(ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING.getKey(), "");
    }

    @SneakyThrows
    public void testSearchResponse_excludesVectorFieldByDefault() {
        String indexName = INDEX_NAME + "-default";
        createIndexWithVectorAndTextField(indexName);
        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "hello world");
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNotNull(source);
        assertTrue(source.containsKey(TEXT_FIELD));
        assertFalse("Vector field should be excluded from _source by default", source.containsKey(VECTOR_FIELD));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_includesVectorFieldWhenExplicitlyRequested() {
        String indexName = INDEX_NAME + "-explicit-include";
        createIndexWithVectorAndTextField(indexName);
        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "hello world");
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("includes", VECTOR_FIELD)
            .endObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNotNull(source);
        assertTrue("Vector field should be present when explicitly included", source.containsKey(VECTOR_FIELD));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_sourceFalse_noSourceReturned() {
        String indexName = INDEX_NAME + "-source-false";
        createIndexWithVectorAndTextField(indexName);
        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "hello world");
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .field("_source", false)
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNull("Source should not be returned when _source is false", source);

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_preservesExistingExcludes() {
        String indexName = INDEX_NAME + "-preserves-excludes";
        createIndexWithVectorAndTextField(indexName);
        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "hello world");
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("excludes", TEXT_FIELD)
            .endObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNotNull(source);
        assertFalse("Text field should be excluded per user request", source.containsKey(TEXT_FIELD));
        assertFalse("Vector field should also be excluded by the processor", source.containsKey(VECTOR_FIELD));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_noVectorField_returnsAllFields() {
        String indexName = INDEX_NAME + "-no-vector";
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEXT_FIELD)
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        String doc = XContentFactory.jsonBuilder().startObject().field(TEXT_FIELD, "hello world").endObject().toString();
        addKnnDoc(indexName, "1", doc);
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNotNull(source);
        assertTrue("Text field should be present when no vector fields in index", source.containsKey(TEXT_FIELD));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_multipleVectorFields_allExcluded() {
        String indexName = INDEX_NAME + "-multi-vector";
        String vectorField2 = "another_vector";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(vectorField2)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(TEXT_FIELD)
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        String doc = XContentFactory.jsonBuilder()
            .startObject()
            .field(VECTOR_FIELD, TEST_VECTOR)
            .field(vectorField2, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f })
            .field(TEXT_FIELD, "multiple vectors")
            .endObject()
            .toString();
        addKnnDoc(indexName, "1", doc);
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNotNull(source);
        assertTrue(source.containsKey(TEXT_FIELD));
        assertFalse("First vector field should be excluded", source.containsKey(VECTOR_FIELD));
        assertFalse("Second vector field should be excluded", source.containsKey(vectorField2));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_nestedVectorField_excluded() {
        String indexName = INDEX_NAME + "-nested-vec";
        String nestedField = "nested_obj";
        String nestedVecField = "vec";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(nestedField)
            .field("type", "nested")
            .startObject("properties")
            .startObject(nestedVecField)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .startObject(TEXT_FIELD)
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        String doc = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(nestedField)
            .field(nestedVecField, TEST_VECTOR)
            .endObject()
            .field(TEXT_FIELD, "nested vector test")
            .endObject()
            .toString();
        addKnnDoc(indexName, "1", doc);
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNotNull(source);
        assertTrue(source.containsKey(TEXT_FIELD));
        if (source.containsKey(nestedField)) {
            Map<String, Object> nestedSource = (Map<String, Object>) source.get(nestedField);
            assertFalse("Nested vector field should be excluded", nestedSource.containsKey(nestedVecField));
        }

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testKnnSearch_excludesVectorFieldByDefault() {
        String indexName = INDEX_NAME + "-knn-search";
        createIndexWithVectorAndTextField(indexName);
        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "document one");
        indexDocumentWithVectorAndText(indexName, "2", new Float[] { 5.0f, 6.0f, 7.0f, 8.0f }, "document two");
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(VECTOR_FIELD)
            .field("vector", TEST_VECTOR)
            .field("k", 2)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertTrue(hits.size() > 0);
        for (Map<String, Object> hit : hits) {
            Map<String, Object> source = getSource(hit);
            assertNotNull(source);
            assertTrue(source.containsKey(TEXT_FIELD));
            assertFalse("Vector field should be excluded from KNN search results", source.containsKey(VECTOR_FIELD));
        }

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_aliasResolvesToConcreteIndex_excludesVectorField() {
        String indexName = INDEX_NAME + "-alias-target";
        String aliasName = "my-test-alias";
        createIndexWithVectorAndTextField(indexName);
        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "alias test");
        refreshIndex(indexName);

        // Create alias pointing to the concrete index
        Request aliasRequest = new Request("POST", "/_aliases");
        aliasRequest.setJsonEntity(
            XContentFactory.jsonBuilder()
                .startObject()
                .startArray("actions")
                .startObject()
                .startObject("add")
                .field("index", indexName)
                .field("alias", aliasName)
                .endObject()
                .endObject()
                .endArray()
                .endObject()
                .toString()
        );
        client().performRequest(aliasRequest);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(aliasName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNotNull(source);
        assertTrue(source.containsKey(TEXT_FIELD));
        assertFalse("Vector field should be excluded when searching via alias", source.containsKey(VECTOR_FIELD));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_nestedQueryWithFetchFields_includesVectorField() {
        String indexName = INDEX_NAME + "-fetch-fields";
        String nestedField = "nested_obj";
        String nestedVecField = "vec";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(nestedField)
            .field("type", "nested")
            .startObject("properties")
            .startObject(nestedVecField)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .startObject(TEXT_FIELD)
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        String doc = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(nestedField)
            .field(nestedVecField, TEST_VECTOR)
            .endObject()
            .field(TEXT_FIELD, "fetch fields test")
            .endObject()
            .toString();
        addKnnDoc(indexName, "1", doc);
        refreshIndex(indexName);

        // Query with nested inner_hits that uses "fields" to fetch the vector field
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", nestedField)
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("inner_hits")
            .startArray("fields")
            .value(nestedField + "." + nestedVecField)
            .endArray()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> hit = hits.get(0);
        Map<String, Object> innerHits = (Map<String, Object>) hit.get("inner_hits");
        assertNotNull(innerHits);
        Map<String, Object> nestedInnerHit = (Map<String, Object>) innerHits.get(nestedField);
        assertNotNull(nestedInnerHit);
        Map<String, Object> innerHitsHits = (Map<String, Object>) nestedInnerHit.get("hits");
        List<Map<String, Object>> innerHitsList = (List<Map<String, Object>>) innerHitsHits.get("hits");
        assertFalse(innerHitsList.isEmpty());

        // The inner hit should have the vector field in "fields" since it was explicitly requested
        Map<String, Object> innerHitFields = (Map<String, Object>) innerHitsList.get(0).get("fields");
        assertNotNull("Inner hit should have fields with vector data", innerHitFields);
        assertTrue(
            "Vector field should be present in inner hit fields when requested via fetch fields",
            innerHitFields.containsKey(nestedField + "." + nestedVecField)
        );

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_storedFieldsNone_vectorFieldNotExcluded() {
        String indexName = INDEX_NAME + "-stored-none";
        createIndexWithVectorAndTextField(indexName);
        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "stored fields test");
        refreshIndex(indexName);

        // When stored_fields is _none_, _source is not returned and the processor should not apply
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .field("stored_fields", "_none_")
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNull("Source should not be returned when stored_fields is _none_", source);

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_processorDisabled_vectorFieldReturned() {
        disableSourceExcludesProcessor();

        String indexName = INDEX_NAME + "-disabled";
        createIndexWithVectorAndTextField(indexName);
        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "hello world");
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> source = getSource(hits.get(0));
        assertNotNull(source);
        assertTrue("Vector field should be present when processor is disabled", source.containsKey(VECTOR_FIELD));
        assertTrue(source.containsKey(TEXT_FIELD));

        deleteIndex(indexName);

        enableSourceExcludesProcessor();
    }

    // Helper methods

    private void createIndexWithVectorAndTextField(String indexName) throws IOException {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(TEXT_FIELD)
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);
    }

    private void indexDocumentWithVectorAndText(String indexName, String docId, Float[] vector, String text) throws IOException {
        String doc = XContentFactory.jsonBuilder().startObject().field(VECTOR_FIELD, vector).field(TEXT_FIELD, text).endObject().toString();
        addKnnDoc(indexName, docId, doc);
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> parseHits(Response response) throws IOException, ParseException {
        String responseBody = EntityUtils.toString(response.getEntity());
        Map<String, Object> responseMap = createParser(JsonXContent.jsonXContent, responseBody).map();
        Map<String, Object> hitsOuter = (Map<String, Object>) responseMap.get("hits");
        return (List<Map<String, Object>>) hitsOuter.get("hits");
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getSource(Map<String, Object> hit) {
        return (Map<String, Object>) hit.get("_source");
    }
}
