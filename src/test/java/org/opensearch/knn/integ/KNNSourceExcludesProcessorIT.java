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
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.script.Script;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringScriptEngine;
import org.opensearch.knn.search.processor.KNNSourceExcludesProcessor;
import org.opensearch.knn.search.processor.mmr.MMROverSampleProcessor;
import org.opensearch.knn.search.processor.mmr.MMRRerankProcessor;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.CANDIDATES;
import static org.opensearch.knn.common.KNNConstants.DIVERSITY;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.MMR;
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
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(vectorField2)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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
    public void testSearchResponse_aliasWithMultipleVectorFieldsAndDocs_excludesAllVectors() {
        String indexName = INDEX_NAME + "-alias-multi";
        String aliasName = "my-test-alias";
        String vectorField2 = "another_vector";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(vectorField2)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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

        String doc1 = XContentFactory.jsonBuilder()
            .startObject()
            .field(VECTOR_FIELD, TEST_VECTOR)
            .field(vectorField2, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f })
            .field(TEXT_FIELD, "document one")
            .endObject()
            .toString();
        addKnnDoc(indexName, "1", doc1);

        String doc2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(VECTOR_FIELD, new Float[] { 9.0f, 10.0f, 11.0f, 12.0f })
            .field(vectorField2, new Float[] { 13.0f, 14.0f, 15.0f, 16.0f })
            .field(TEXT_FIELD, "document two")
            .endObject()
            .toString();
        addKnnDoc(indexName, "2", doc2);

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

        assertEquals(2, hits.size());
        for (Map<String, Object> hit : hits) {
            Map<String, Object> source = getSource(hit);
            assertNotNull(source);
            assertTrue(source.containsKey(TEXT_FIELD));
            assertFalse("First vector field should be excluded when searching via alias", source.containsKey(VECTOR_FIELD));
            assertFalse("Second vector field should be excluded when searching via alias", source.containsKey(vectorField2));
        }

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
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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

    @SneakyThrows
    public void testSearchResponse_sourceDisabledAtIndexLevel_noSourceReturned() {
        String indexName = INDEX_NAME + "-source-disabled";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .field("enabled", false)
            .endObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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
        assertNull("_source should not be returned when disabled at index level", getSource(hits.get(0)));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_vectorFieldExcludedByIndexMappingLiteral_processorDoesNotDuplicate() {
        String indexName = INDEX_NAME + "-mapping-exclude-literal";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("excludes", VECTOR_FIELD)
            .endObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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
        assertFalse("Vector field should be absent because mapping excludes it", source.containsKey(VECTOR_FIELD));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_vectorFieldExcludedByIndexMappingGlob_processorDoesNotDuplicate() {
        String indexName = INDEX_NAME + "-mapping-exclude-glob";
        String globPattern = "my_*";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("_source")
            .array("excludes", globPattern)
            .endObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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
        assertFalse("Vector field matching glob should be absent", source.containsKey(VECTOR_FIELD));
        assertTrue("Non-matching field should still be present", source.containsKey(TEXT_FIELD));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_innerHitExplicitTrueSource_noTopLevelExcludes() {
        String indexName = INDEX_NAME + "-inner-hit-true";
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
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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
            .field(TEXT_FIELD, "inner hit true source test")
            .endObject()
            .toString();
        addKnnDoc(indexName, "1", doc);
        refreshIndex(indexName);

        // Inner hit with _source: true — processor should not add top-level excludes (issue #3303)
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
            .field("_source", true)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(1, hits.size());
        Map<String, Object> hit = hits.get(0);

        // Top-level source should contain the nested object (vector not excluded at top level)
        Map<String, Object> source = getSource(hit);
        assertNotNull(source);
        assertTrue("Nested object should be present in top-level source", source.containsKey(nestedField));

        // Inner hit source should contain the vector
        Map<String, Object> innerHits = (Map<String, Object>) hit.get("inner_hits");
        assertNotNull(innerHits);
        Map<String, Object> nestedInnerHit = (Map<String, Object>) innerHits.get(nestedField);
        Map<String, Object> innerHitsHits = (Map<String, Object>) nestedInnerHit.get("hits");
        List<Map<String, Object>> innerHitsList = (List<Map<String, Object>>) innerHitsHits.get("hits");
        assertFalse(innerHitsList.isEmpty());
        Map<String, Object> innerSource = (Map<String, Object>) innerHitsList.get(0).get("_source");
        assertNotNull("Inner hit should have _source", innerSource);
        assertTrue("Vector field should be present in inner hit _source", innerSource.containsKey(nestedVecField));

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_mmrAndSourceExcludesProcessorsTogether_vectorExcludedAndMMRReranks() {
        // Enable both MMR and source-excludes processors simultaneously
        updateClusterSettings(
            ENABLED_SYSTEM_GENERATED_FACTORIES_SETTING.getKey(),
            new String[] {
                KNNSourceExcludesProcessor.Factory.TYPE,
                MMROverSampleProcessor.MMROverSampleProcessorFactory.TYPE,
                MMRRerankProcessor.MMRRerankProcessorFactory.TYPE }
        );

        String indexName = INDEX_NAME + "-mmr-tandem";
        String textField2 = "description";

        // Use FAISS since MMR tests use it; 2D vectors for simplicity
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", 2)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, BuiltinKNNEngine.FAISS.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .startObject(TEXT_FIELD)
            .field("type", "text")
            .endObject()
            .startObject(textField2)
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        // Index 10 docs: 8 similar + 2 diverse
        for (int i = 0; i < 8; i++) {
            String doc = XContentFactory.jsonBuilder()
                .startObject()
                .field(VECTOR_FIELD, new float[] { 1.0f, 1.0f })
                .field(TEXT_FIELD, "similar doc " + i)
                .field(textField2, "description " + i)
                .endObject()
                .toString();
            addKnnDoc(indexName, String.valueOf(i), doc);
        }
        addKnnDoc(
            indexName,
            "8",
            XContentFactory.jsonBuilder()
                .startObject()
                .field(VECTOR_FIELD, new float[] { 1.0f, 2.0f })
                .field(TEXT_FIELD, "diverse doc 8")
                .field(textField2, "description 8")
                .endObject()
                .toString()
        );
        addKnnDoc(
            indexName,
            "9",
            XContentFactory.jsonBuilder()
                .startObject()
                .field(VECTOR_FIELD, new float[] { 2.0f, 1.0f })
                .field(TEXT_FIELD, "diverse doc 9")
                .field(textField2, "description 9")
                .endObject()
                .toString()
        );

        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .field("size", 3)
            .startObject("query")
            .startObject(KNN)
            .startObject(VECTOR_FIELD)
            .array("vector", new float[] { 1.0f, 1.0f })
            .field(K, 10)
            .endObject()
            .endObject()
            .endObject()
            .startObject("ext")
            .startObject(MMR)
            .field(CANDIDATES, 10)
            .field(DIVERSITY, 0.9)
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals("MMR should return 3 results", 3, hits.size());

        List<String> returnedIds = hits.stream().map(hit -> (String) hit.get("_id")).toList();
        assertTrue("MMR with high diversity should select diverse doc 8", returnedIds.contains("8"));
        assertTrue("MMR with high diversity should select diverse doc 9", returnedIds.contains("9"));

        for (Map<String, Object> hit : hits) {
            Map<String, Object> source = getSource(hit);
            assertNotNull("_source should be present", source);
            assertFalse("Vector field should be excluded by source-excludes processor", source.containsKey(VECTOR_FIELD));
            assertTrue("Text field should be present", source.containsKey(TEXT_FIELD));
            assertTrue("Description field should be present", source.containsKey(textField2));
        }

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_knnScoreScriptExactSearch_withPreFilter_vectorExcluded() {
        String indexName = INDEX_NAME + "-knn-score-script";

        // index.knn: false — exact search via knn_score script, no ANN graph built
        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", false).build();

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .endObject()
            .startObject(TEXT_FIELD)
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, settings, mapping);

        indexDocumentWithVectorAndText(indexName, "1", new Float[] { 1.0f, 2.0f, 3.0f, 4.0f }, "relevant");
        indexDocumentWithVectorAndText(indexName, "2", new Float[] { 2.0f, 3.0f, 4.0f, 5.0f }, "relevant");
        indexDocumentWithVectorAndText(indexName, "3", new Float[] { 10.0f, 10.0f, 10.0f, 10.0f }, "irrelevant");
        refreshIndex(indexName);

        // Use knn_score script (lang: knn) with a pre-filter on TEXT_FIELD to narrow candidates —
        // the documented pattern for exact k-NN with filtering.
        // Score is: 1 / (1 + l2_distance) for L2 space.
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .field("size", 2)
            .startObject("query")
            .startObject("script_score")
            .startObject("query")
            .startObject("term")
            .field(TEXT_FIELD, "relevant")
            .endObject()
            .endObject()
            .startObject("script")
            .field("source", KNNScoringScriptEngine.SCRIPT_SOURCE)
            .field("lang", KNNScoringScriptEngine.NAME)
            .startObject("params")
            .field("field", VECTOR_FIELD)
            .array("query_value", new float[] { 1.0f, 2.0f, 3.0f, 4.0f })
            .field("space_type", SpaceType.L2.getValue())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Response response = performSearch(indexName, query);
        List<Map<String, Object>> hits = parseHits(response);

        // Pre-filter narrows to "relevant" docs — doc 3 is excluded
        assertEquals("Only relevant docs should be returned", 2, hits.size());

        for (Map<String, Object> hit : hits) {
            Map<String, Object> source = getSource(hit);
            assertNotNull("_source should be present", source);
            assertFalse(
                "Vector field should be excluded by processor even with knn_score script exact search",
                source.containsKey(VECTOR_FIELD)
            );
            assertEquals("Only relevant docs returned", "relevant", source.get(TEXT_FIELD));

            // L2 space score formula: 1 / (1 + l2Squared(query, doc))
            // l2Squared is the sum of squared differences per dimension — not the actual L2 norm.
            double score = ((Number) hit.get("_score")).doubleValue();
            assertTrue("Score should be positive", score > 0.0);
            assertTrue("Score should be at most 1.0 (when l2Squared=0)", score <= 1.0 + 1e-6);
        }

        // Doc 1 == query vector → l2Squared = 0 → score = 1/(1+0) = 1.0 (highest)
        assertEquals("Doc 1 (l2Squared=0) should rank first", "1", hits.get(0).get("_id"));
        assertEquals(1.0, ((Number) hits.get(0).get("_score")).doubleValue(), 1e-6);

        // Doc 2 = [2,3,4,5], query = [1,2,3,4] → l2Squared = 4 → score = 1/(1+4) = 0.2
        assertEquals("Doc 2 should rank second", "2", hits.get(1).get("_id"));
        assertEquals(0.2, ((Number) hits.get(1).get("_score")).doubleValue(), 1e-6);

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testSearchResponse_painlessScriptScoringExactSearch_indexKnnFalse_vectorExcluded() {
        String indexName = INDEX_NAME + "-script-score";

        // index.knn: false — no ANN graph; vectors are accessible via doc values for script scoring
        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", false).build();

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", TYPE_KNN_VECTOR)
            .field("dimension", DIMENSION_VALUE)
            .endObject()
            .startObject(TEXT_FIELD)
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, settings, mapping);

        // Doc 1: identical to query vector — distance 0, highest score
        // Doc 2: slightly different — small distance
        // Doc 3: far away — large distance, lowest score
        indexDocumentWithVectorAndText(indexName, "1", new Float[] { 1.0f, 2.0f, 3.0f, 4.0f }, "doc one");
        indexDocumentWithVectorAndText(indexName, "2", new Float[] { 2.0f, 3.0f, 4.0f, 5.0f }, "doc two");
        indexDocumentWithVectorAndText(indexName, "3", new Float[] { 10.0f, 10.0f, 10.0f, 10.0f }, "doc three far");
        refreshIndex(indexName);

        // Painless script: standard L2 score boosted by a constant multiplier (2.0) to manipulate the distance computation.
        // score = (1 / (1 + l2Squared(queryVec, doc[field]))) * 2.0
        String scriptSource = String.format("(1.0 / (1.0 + l2Squared([1.0f, 2.0f, 3.0f, 4.0f], doc['%s']))) * 2.0", VECTOR_FIELD);

        Request request = constructScriptScoreContextSearchRequest(
            indexName,
            new MatchAllQueryBuilder(),
            new HashMap<>(),
            Script.DEFAULT_SCRIPT_LANG,
            scriptSource,
            3,
            new HashMap<>()
        );

        Response response = client().performRequest(request);
        List<Map<String, Object>> hits = parseHits(response);

        assertEquals(3, hits.size());

        // Doc 1 is identical to query — l2Squared = 0 → score = 2.0 (highest)
        assertEquals("Doc 1 (l2=0) should rank first", "1", hits.get(0).get("_id"));

        for (Map<String, Object> hit : hits) {
            Map<String, Object> source = getSource(hit);
            assertNotNull("_source should be present", source);
            assertFalse(
                "Vector field should be excluded by source-excludes processor even with script scoring on index.knn=false",
                source.containsKey(VECTOR_FIELD)
            );
            assertTrue("Text field should be present", source.containsKey(TEXT_FIELD));

            // Verify the script manipulation: all scores should be > 0 and ≤ 2.0 (the max with l2=0)
            double score = ((Number) hit.get("_score")).doubleValue();
            assertTrue("Score should be positive", score > 0.0);
            assertTrue("Score should be at most 2.0 (boosted score when l2=0)", score <= 2.0 + 1e-6);
        }

        deleteIndex(indexName);
    }

    @SneakyThrows
    public void testReindex_withProcessorEnabled_vectorReindexedAndExcludedInDestination() {
        String sourceIndex = INDEX_NAME + "-reindex-source";
        String destIndex = INDEX_NAME + "-reindex-dest";

        createIndexWithVectorAndTextField(sourceIndex);
        indexDocumentWithVectorAndText(sourceIndex, "1", TEST_VECTOR, "doc one");
        indexDocumentWithVectorAndText(sourceIndex, "2", new Float[] { 5.0f, 6.0f, 7.0f, 8.0f }, "doc two");
        refreshIndex(sourceIndex);

        // Create destination index with same mapping
        createIndexWithVectorAndTextField(destIndex);

        // Reindex — internal search of source must not be affected by the processor
        reindex(sourceIndex, destIndex);
        refreshIndex(destIndex);

        // Verify all docs were reindexed with vectors intact via _get (bypasses search pipeline)
        for (String docId : new String[] { "1", "2" }) {
            Map<String, Object> source = getKnnDoc(destIndex, docId);
            assertNotNull("Doc " + docId + " should exist in destination", source);
            assertTrue("Vector should be stored in destination doc " + docId, source.containsKey(VECTOR_FIELD));
            assertTrue("Text field should be present in destination doc " + docId, source.containsKey(TEXT_FIELD));
        }

        // Search destination — processor should exclude vectors from search response
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        List<Map<String, Object>> hits = parseHits(performSearch(destIndex, query));
        assertEquals(2, hits.size());
        for (Map<String, Object> hit : hits) {
            Map<String, Object> hitSource = getSource(hit);
            assertNotNull(hitSource);
            assertTrue("Text field should be present in search response", hitSource.containsKey(TEXT_FIELD));
            assertFalse("Vector should be excluded from search response in destination", hitSource.containsKey(VECTOR_FIELD));
        }

        deleteIndex(sourceIndex);
        deleteIndex(destIndex);
    }

    @SneakyThrows
    public void testUpdateByQuery_withProcessorEnabled_updatesSucceedAndVectorStillExcluded() {
        String indexName = INDEX_NAME + "-update-by-query";
        createIndexWithVectorAndTextField(indexName);

        indexDocumentWithVectorAndText(indexName, "1", TEST_VECTOR, "original text one");
        indexDocumentWithVectorAndText(indexName, "2", TEST_VECTOR, "original text two");
        refreshIndex(indexName);

        // Run update_by_query — its internal search must not be affected by the processor
        String updateScript = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("script")
            .field("source", "ctx._source." + TEXT_FIELD + " = 'updated text'")
            .field("lang", "painless")
            .endObject()
            .endObject()
            .toString();

        updateKnnDocByQuery(indexName, updateScript);

        // Use _get to verify the update succeeded — bypasses the search pipeline entirely
        for (String docId : new String[] { "1", "2" }) {
            Map<String, Object> source = getKnnDoc(indexName, docId);
            assertNotNull("Doc " + docId + " should exist", source);
            assertEquals("Doc " + docId + " text field should be updated", "updated text", source.get(TEXT_FIELD));
            // _get returns full source — vector should still be stored (update didn't remove it)
            assertTrue("Vector should still be stored in doc " + docId, source.containsKey(VECTOR_FIELD));
        }

        deleteIndex(indexName);
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
            .field(KNN_ENGINE, BuiltinKNNEngine.LUCENE.getName())
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
