/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * Integration tests for the docvalue_fields feature on knn_vector fields.
 * Validates that vectors can be retrieved directly from doc values in search responses
 * across different engines, data types, query types, and field configurations.
 */
public class DocValueFieldsIT extends KNNRestTestCase {

    private static final String TEST_INDEX = "docvalue_fields_test_index";
    private static final String VECTOR_FIELD = "test_vector";
    private static final int DIMENSION = 4;
    private static final float[] VECTOR_1 = { 1.0f, 2.0f, 3.0f, 4.0f };
    private static final float[] VECTOR_2 = { 5.0f, 6.0f, 7.0f, 8.0f };
    private static final float[] VECTOR_3 = { 9.0f, 10.0f, 11.0f, 12.0f };

    /**
     * Verifies that docvalue_fields returns correct vectors for both Faiss and Lucene HNSW indexes
     * when _source is disabled. Validates each doc ID's returned vector matches its indexed vector.
     */
    @SneakyThrows
    public void testDocValueFields_faissAndLuceneHnsw_returnsCorrectVectorsWithoutSource() {
        for (KNNEngine engine : List.of(KNNEngine.FAISS, KNNEngine.LUCENE)) {
            String indexName = TEST_INDEX + "_" + engine.getName();
            createHnswIndex(indexName, engine);
            indexTestDocuments(indexName);
            forceMergeKnnIndex(indexName);

            String query = buildSortedDocValueFieldsQuery();
            Response response = searchKNNIndex(indexName, query, 10);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<Map<String, Object>> hits = parseSearchHits(responseBody);

            assertEquals("[" + engine.getName() + "] Expected 3 hits", 3, hits.size());

            Map<String, List<Double>> docIdToVector = new java.util.LinkedHashMap<>();
            for (Map<String, Object> hit : hits) {
                assertNull("[" + engine.getName() + "] _source should be null", hit.get("_source"));
                assertNotNull("[" + engine.getName() + "] fields should be present", hit.get("fields"));
                String docId = (String) hit.get("_id");
                List<List<Double>> vectorField = getDocValueField(hit, VECTOR_FIELD);
                assertNotNull("[" + engine.getName() + "] vector field should not be null for doc " + docId, vectorField);
                assertFalse("[" + engine.getName() + "] vector field should not be empty for doc " + docId, vectorField.isEmpty());
                assertEquals("[" + engine.getName() + "] dimension mismatch for doc " + docId, DIMENSION, vectorField.get(0).size());
                docIdToVector.put(docId, vectorField.get(0));
            }

            assertVectorForDoc(docIdToVector, "1", VECTOR_1, engine.getName());
            assertVectorForDoc(docIdToVector, "2", VECTOR_2, engine.getName());
            assertVectorForDoc(docIdToVector, "3", VECTOR_3, engine.getName());

            deleteKNNIndex(indexName);
        }
    }

    /**
     * Verifies that vector values returned via docvalue_fields exactly match
     * those returned via _source for all indexed documents.
     */
    @SneakyThrows
    public void testDocValueFields_vectorValuesMatchSource() {
        createHnswIndex(KNNEngine.FAISS);
        indexTestDocuments();

        // Fetch all docs with _source
        String sourceQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();

        Response sourceResponse = searchKNNIndex(TEST_INDEX, sourceQuery, 10);
        String sourceBody = EntityUtils.toString(sourceResponse.getEntity());
        List<Map<String, Object>> sourceHits = parseSearchHits(sourceBody);

        // Fetch all docs with docvalue_fields
        String dvQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", true)
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();

        Response dvResponse = searchKNNIndex(TEST_INDEX, dvQuery, 10);
        String dvBody = EntityUtils.toString(dvResponse.getEntity());
        List<Map<String, Object>> dvHits = parseSearchHits(dvBody);

        assertEquals(3, sourceHits.size());
        assertEquals(3, dvHits.size());

        // Collect all source and docvalue vectors
        List<List<Double>> sourceVectors = new ArrayList<>();
        List<List<Double>> dvVectors = new ArrayList<>();
        for (int i = 0; i < sourceHits.size(); i++) {
            sourceVectors.add(getSourceVector(sourceHits.get(i), VECTOR_FIELD));
            List<List<Double>> dvField = getDocValueField(dvHits.get(i), VECTOR_FIELD);
            assertNotNull(dvField);
            dvVectors.add(dvField.get(0));
        }

        // Validate all vectors match
        for (int i = 0; i < sourceVectors.size(); i++) {
            List<Double> fromSource = sourceVectors.get(i);
            List<Double> fromDocValues = dvVectors.get(i);
            assertEquals("Dimension mismatch for doc " + i, fromSource.size(), fromDocValues.size());
            for (int j = 0; j < fromSource.size(); j++) {
                assertEquals("Value mismatch at index " + j + " for doc " + i, fromSource.get(j), fromDocValues.get(j), 0.001);
            }
        }
    }

    /**
     * Verifies that docvalue_fields works correctly with a KNN query returning top-K results,
     * and each hit's vector matches the vector indexed for that doc ID.
     */
    @SneakyThrows
    public void testDocValueFields_withKnnQuery_topK() {
        createHnswIndex(KNNEngine.FAISS);
        indexTestDocuments();

        String query = buildDocValueFieldsQuery(VECTOR_1, false);
        Response response = searchKNNIndex(TEST_INDEX, query, 3);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(3, hits.size());

        Map<String, List<Double>> docIdToVector = new java.util.LinkedHashMap<>();
        for (Map<String, Object> hit : hits) {
            String docId = (String) hit.get("_id");
            List<List<Double>> vectorField = getDocValueField(hit, VECTOR_FIELD);
            assertNotNull(vectorField);
            assertEquals(1, vectorField.size());
            assertEquals(DIMENSION, vectorField.get(0).size());
            docIdToVector.put(docId, vectorField.get(0));
        }

        assertVectorForDoc(docIdToVector, "1", VECTOR_1);
        assertVectorForDoc(docIdToVector, "2", VECTOR_2);
        assertVectorForDoc(docIdToVector, "3", VECTOR_3);
    }

    /**
     * Verifies that docvalue_fields works with a match_all query, returning vectors
     * for all documents in the index with correct values per doc ID.
     */
    @SneakyThrows
    public void testDocValueFields_withMatchAllQuery() {
        createHnswIndex(KNNEngine.FAISS);
        indexTestDocuments();

        String query = buildSortedDocValueFieldsQuery();
        Response response = searchKNNIndex(TEST_INDEX, query, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(3, hits.size());

        Map<String, List<Double>> docIdToVector = new java.util.LinkedHashMap<>();
        for (Map<String, Object> hit : hits) {
            String docId = (String) hit.get("_id");
            List<List<Double>> vectorField = getDocValueField(hit, VECTOR_FIELD);
            assertNotNull(vectorField);
            assertEquals(DIMENSION, vectorField.get(0).size());
            docIdToVector.put(docId, vectorField.get(0));
        }

        assertVectorForDoc(docIdToVector, "1", VECTOR_1);
        assertVectorForDoc(docIdToVector, "2", VECTOR_2);
        assertVectorForDoc(docIdToVector, "3", VECTOR_3);
    }

    /**
     * Verifies that docvalue_fields correctly returns high-dimensional vectors (768d)
     * with all components matching the indexed values.
     */
    @SneakyThrows
    public void testDocValueFields_highDimension_768d() {
        int highDimension = 768;
        String indexName = TEST_INDEX + "_high_dim";

        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.L2.getValue())
            .engine(KNNEngine.FAISS.getName())
            .build();

        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(VECTOR_FIELD)
            .dimension(highDimension)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        createKnnIndex(indexName, mapping);

        float[] vector768 = new float[highDimension];
        for (int i = 0; i < highDimension; i++) {
            vector768[i] = (float) (i * 0.01);
        }
        addKnnDoc(indexName, "1", VECTOR_FIELD, Floats.asList(vector768).toArray());
        refreshIndex(indexName);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(VECTOR_FIELD)
            .field("vector", vector768)
            .field("k", 1)
            .endObject()
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(indexName, query, 1);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(1, hits.size());
        List<List<Double>> vectorField = getDocValueField(hits.get(0), VECTOR_FIELD);
        assertNotNull(vectorField);
        assertEquals(highDimension, vectorField.get(0).size());
        for (int i = 0; i < highDimension; i++) {
            assertEquals(vector768[i], vectorField.get(0).get(i).floatValue(), 0.001f);
        }

        deleteKNNIndex(indexName);
    }

    /**
     * Verifies that docvalue_fields can return multiple vector fields simultaneously
     * when an index contains more than one knn_vector field.
     */
    @SneakyThrows
    public void testDocValueFields_multipleVectorFields() {
        String secondVectorField = "second_vector";
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject(secondVectorField)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(TEST_INDEX, mapping);

        XContentBuilder docBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(VECTOR_FIELD, VECTOR_1)
            .field(secondVectorField, VECTOR_2)
            .endObject();
        Request request = new Request("POST", "/" + TEST_INDEX + "/_doc/1?refresh=true");
        request.setJsonEntity(docBuilder.toString());
        client().performRequest(request);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startArray("docvalue_fields")
            .value(VECTOR_FIELD)
            .value(secondVectorField)
            .endArray()
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(TEST_INDEX, query, 1);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(1, hits.size());
        List<List<Double>> firstVector = getDocValueField(hits.get(0), VECTOR_FIELD);
        List<List<Double>> secondVector = getDocValueField(hits.get(0), secondVectorField);
        assertNotNull(firstVector);
        assertNotNull(secondVector);
        assertEquals(DIMENSION, firstVector.get(0).size());
        assertEquals(DIMENSION, secondVector.get(0).size());
    }

    /**
     * Verifies that docvalue_fields can return vector fields alongside scalar fields
     * (e.g., long) in the same search response.
     */
    @SneakyThrows
    public void testDocValueFields_mixedWithScalarFields() {
        String numericField = "price";
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject(numericField)
            .field("type", "long")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(TEST_INDEX, mapping);
        addKnnDocWithNumericField(TEST_INDEX, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray(), numericField, 100L);
        refreshIndex(TEST_INDEX);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startArray("docvalue_fields")
            .value(VECTOR_FIELD)
            .value(numericField)
            .endArray()
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(TEST_INDEX, query, 1);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(1, hits.size());
        Map<String, Object> fields = getFields(hits.get(0));
        assertNotNull(fields.get(VECTOR_FIELD));
        assertNotNull(fields.get(numericField));
    }

    /**
     * Verifies that docvalue_fields and _source can be returned together in the same response
     * when _source is not explicitly disabled.
     */
    @SneakyThrows
    public void testDocValueFields_withSourceEnabled() {
        createHnswIndex(KNNEngine.FAISS);
        addKnnDoc(TEST_INDEX, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        refreshIndex(TEST_INDEX);

        String query = buildDocValueFieldsQuery(VECTOR_1, true);
        Response response = searchKNNIndex(TEST_INDEX, query, 1);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(1, hits.size());
        assertNotNull(hits.get(0).get("_source"));
        List<List<Double>> vectorField = getDocValueField(hits.get(0), VECTOR_FIELD);
        assertNotNull(vectorField);
    }

    /**
     * Verifies that when _source is explicitly disabled, only the doc values vector
     * is present in the response (no _source field).
     */
    @SneakyThrows
    public void testDocValueFields_sourceDisabled_onlyDocValues() {
        createHnswIndex(KNNEngine.FAISS);
        addKnnDoc(TEST_INDEX, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        refreshIndex(TEST_INDEX);

        String query = buildDocValueFieldsQuery(VECTOR_1, false);
        Response response = searchKNNIndex(TEST_INDEX, query, 1);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(1, hits.size());
        assertNull(hits.get(0).get("_source"));
        List<List<Double>> vectorField = getDocValueField(hits.get(0), VECTOR_FIELD);
        assertNotNull(vectorField);
        assertEquals(DIMENSION, vectorField.get(0).size());
    }

    /**
     * Verifies that querying an empty index with docvalue_fields returns an empty hits array
     * without errors.
     */
    @SneakyThrows
    public void testDocValueFields_emptyIndex_returnsEmptyHits() {
        createHnswIndex(KNNEngine.FAISS);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(TEST_INDEX, query, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertTrue(hits.isEmpty());
    }

    /**
     * Verifies that after deleting a document, its vector is no longer returned
     * via docvalue_fields (no ghost vectors from deleted docs).
     */
    @SneakyThrows
    public void testDocValueFields_afterDocDeletion_noGhostVectors() {
        createHnswIndex(KNNEngine.FAISS);
        addKnnDoc(TEST_INDEX, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        addKnnDoc(TEST_INDEX, "2", VECTOR_FIELD, Floats.asList(VECTOR_2).toArray());
        refreshIndex(TEST_INDEX);

        deleteKnnDoc(TEST_INDEX, "1");
        refreshIndex(TEST_INDEX);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(TEST_INDEX, query, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(1, hits.size());
        assertEquals("2", hits.get(0).get("_id"));
        List<List<Double>> vectorField = getDocValueField(hits.get(0), VECTOR_FIELD);
        assertNotNull(vectorField);
    }

    /**
     * Verifies that docvalue_fields returns correct vectors after a force merge operation,
     * ensuring segment merging does not corrupt vector doc values.
     */
    @SneakyThrows
    public void testDocValueFields_afterForcemerge_returnsCorrectly() {
        createHnswIndex(KNNEngine.FAISS);
        addKnnDoc(TEST_INDEX, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        addKnnDoc(TEST_INDEX, "2", VECTOR_FIELD, Floats.asList(VECTOR_2).toArray());
        addKnnDoc(TEST_INDEX, "3", VECTOR_FIELD, Floats.asList(VECTOR_3).toArray());
        refreshIndex(TEST_INDEX);

        forceMergeKnnIndex(TEST_INDEX);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(TEST_INDEX, query, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(3, hits.size());
        for (Map<String, Object> hit : hits) {
            List<List<Double>> vectorField = getDocValueField(hit, VECTOR_FIELD);
            assertNotNull(vectorField);
            assertEquals(DIMENSION, vectorField.get(0).size());
        }
    }

    /**
     * Verifies that docvalue_fields works correctly with pagination (from/size),
     * returning vectors for all hits in the requested page.
     */
    @SneakyThrows
    public void testDocValueFields_paginationWithFrom() {
        createHnswIndex(KNNEngine.FAISS);
        for (int i = 0; i < 10; i++) {
            float[] vec = new float[] { i * 1.0f, i * 2.0f, i * 3.0f, i * 4.0f };
            addKnnDoc(TEST_INDEX, String.valueOf(i), VECTOR_FIELD, Floats.asList(vec).toArray());
        }
        refreshIndex(TEST_INDEX);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", false)
            .field("from", 5)
            .field("size", 5)
            .endObject()
            .toString();

        Response response = performSearch(TEST_INDEX, query);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(5, hits.size());
        for (Map<String, Object> hit : hits) {
            List<List<Double>> vectorField = getDocValueField(hit, VECTOR_FIELD);
            assertNotNull(vectorField);
            assertEquals(DIMENSION, vectorField.get(0).size());
        }
    }

    /**
     * Verifies that requesting docvalue_fields on a BYTE vector field throws an error
     * for both Faiss and Lucene engines, since only FLOAT vectors are supported.
     */
    @SneakyThrows
    public void testDocValueFields_byteVectorField_throwsError() {
        for (KNNEngine engine : List.of(KNNEngine.FAISS, KNNEngine.LUCENE)) {
            String indexName = TEST_INDEX + "_byte_" + engine.getName();
            KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
                .methodName(METHOD_HNSW)
                .spaceType(SpaceType.L2.getValue())
                .engine(engine.getName())
                .build();

            String mapping = KNNJsonIndexMappingsBuilder.builder()
                .fieldName(VECTOR_FIELD)
                .dimension(DIMENSION)
                .vectorDataType(VectorDataType.BYTE.getValue())
                .method(method)
                .build()
                .getIndexMapping();

            createKnnIndex(indexName, mapping);
            addKnnDoc(indexName, "1", VECTOR_FIELD, new Byte[] { 1, 2, 3, 4 });
            refreshIndex(indexName);

            String query = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("match_all")
                .endObject()
                .endObject()
                .array("docvalue_fields", VECTOR_FIELD)
                .field("_source", false)
                .endObject()
                .toString();

            ResponseException ex = expectThrows(ResponseException.class, () -> searchKNNIndex(indexName, query, 1));
            assertTrue("Expected error for engine " + engine.getName(), ex.getMessage().contains("docvalue_fields is not supported"));
            assertTrue("Expected BYTE in error for engine " + engine.getName(), ex.getMessage().contains("BYTE"));

            deleteKNNIndex(indexName);
        }
    }

    /**
     * Verifies that requesting docvalue_fields on a BINARY vector field throws an error
     * for both Faiss and Lucene engines, since only FLOAT vectors are supported.
     */
    @SneakyThrows
    public void testDocValueFields_binaryVectorField_throwsError() {
        int binaryDimension = 8;
        for (KNNEngine engine : List.of(KNNEngine.FAISS, KNNEngine.LUCENE)) {
            String indexName = TEST_INDEX + "_binary_" + engine.getName();
            KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
                .methodName(METHOD_HNSW)
                .engine(engine.getName())
                .build();

            String mapping = KNNJsonIndexMappingsBuilder.builder()
                .fieldName(VECTOR_FIELD)
                .dimension(binaryDimension)
                .vectorDataType(VectorDataType.BINARY.getValue())
                .method(method)
                .build()
                .getIndexMapping();

            createKnnIndex(indexName, mapping);
            addKnnDoc(indexName, "1", VECTOR_FIELD, new Byte[] { 42 });
            refreshIndex(indexName);

            String query = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("match_all")
                .endObject()
                .endObject()
                .array("docvalue_fields", VECTOR_FIELD)
                .field("_source", false)
                .endObject()
                .toString();

            ResponseException ex = expectThrows(ResponseException.class, () -> searchKNNIndex(indexName, query, 1));
            assertTrue("Expected error for engine " + engine.getName(), ex.getMessage().contains("docvalue_fields is not supported"));
            assertTrue("Expected BINARY in error for engine " + engine.getName(), ex.getMessage().contains("BINARY"));

            deleteKNNIndex(indexName);
        }
    }

    /**
     * Verifies that docvalue_fields returns vectors matching _source when the KNN-specific
     * derived source feature (index.knn.derived_source.enabled) is enabled.
     */
    @SneakyThrows
    public void testDocValueFields_knnDerivedSourceEnabled_vectorsMatchSource() {
        String indexName = TEST_INDEX + "_knn_derived";

        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.L2.getValue())
            .engine(KNNEngine.FAISS.getName())
            .build();

        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(VECTOR_FIELD)
            .dimension(DIMENSION)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, true)
            .build();

        createKnnIndex(indexName, settings, mapping);
        addKnnDoc(indexName, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        addKnnDoc(indexName, "2", VECTOR_FIELD, Floats.asList(VECTOR_2).toArray());
        refreshIndex(indexName);

        // Fetch with _source to get baseline vectors
        String sourceQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();

        Response sourceResponse = searchKNNIndex(indexName, sourceQuery, 2);
        String sourceBody = EntityUtils.toString(sourceResponse.getEntity());
        List<Map<String, Object>> sourceHits = parseSearchHits(sourceBody);

        // Fetch with docvalue_fields
        String dvQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", false)
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();

        Response dvResponse = searchKNNIndex(indexName, dvQuery, 2);
        String dvBody = EntityUtils.toString(dvResponse.getEntity());
        List<Map<String, Object>> dvHits = parseSearchHits(dvBody);

        assertEquals(sourceHits.size(), dvHits.size());
        for (int i = 0; i < sourceHits.size(); i++) {
            List<Double> fromSource = getSourceVector(sourceHits.get(i), VECTOR_FIELD);
            List<List<Double>> fromDocValues = getDocValueField(dvHits.get(i), VECTOR_FIELD);
            assertNotNull(fromDocValues);
            assertEquals(fromSource.size(), fromDocValues.get(0).size());
            for (int j = 0; j < fromSource.size(); j++) {
                assertEquals(fromSource.get(j), fromDocValues.get(0).get(j), 0.001);
            }
        }

        deleteKNNIndex(indexName);
    }

    /**
     * Verifies that docvalue_fields returns vectors matching _source when the core
     * derived source feature (index.derived_source) is enabled.
     */
    @SneakyThrows
    public void testDocValueFields_coreDerivedSourceEnabled_vectorsMatchSource() {
        String indexName = TEST_INDEX + "_core_derived";

        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.L2.getValue())
            .engine(KNNEngine.FAISS.getName())
            .build();

        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(VECTOR_FIELD)
            .dimension(DIMENSION)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(IndexSettings.INDEX_DERIVED_SOURCE_SETTING.getKey(), true)
            .put(IndexSettings.INDEX_DERIVED_SOURCE_TRANSLOG_ENABLED_SETTING.getKey(), true)
            .build();

        createKnnIndex(indexName, settings, mapping);
        addKnnDoc(indexName, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        addKnnDoc(indexName, "2", VECTOR_FIELD, Floats.asList(VECTOR_2).toArray());
        refreshIndex(indexName);

        // Fetch with _source to get baseline vectors (reconstructed from derived source)
        String sourceQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();

        Response sourceResponse = searchKNNIndex(indexName, sourceQuery, 2);
        String sourceBody = EntityUtils.toString(sourceResponse.getEntity());
        List<Map<String, Object>> sourceHits = parseSearchHits(sourceBody);

        // Fetch with docvalue_fields (reads directly from doc values, not derived source)
        String dvQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", false)
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();

        Response dvResponse = searchKNNIndex(indexName, dvQuery, 2);
        String dvBody = EntityUtils.toString(dvResponse.getEntity());
        List<Map<String, Object>> dvHits = parseSearchHits(dvBody);

        assertEquals(sourceHits.size(), dvHits.size());
        for (int i = 0; i < sourceHits.size(); i++) {
            List<Double> fromSource = getSourceVector(sourceHits.get(i), VECTOR_FIELD);
            List<List<Double>> fromDocValues = getDocValueField(dvHits.get(i), VECTOR_FIELD);
            assertNotNull(fromDocValues);
            assertEquals(fromSource.size(), fromDocValues.get(0).size());
            for (int j = 0; j < fromSource.size(); j++) {
                assertEquals(fromSource.get(j), fromDocValues.get(0).get(j), 0.001);
            }
        }

        deleteKNNIndex(indexName);
    }

    /**
     * Verifies that specifying a custom format (e.g., "epoch_millis") for a knn_vector
     * docvalue_fields request results in an error, since vectors do not support custom formats.
     */
    @SneakyThrows
    public void testDocValueFields_customFormat_throwsError() {
        createHnswIndex(KNNEngine.FAISS);
        addKnnDoc(TEST_INDEX, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        refreshIndex(TEST_INDEX);

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startArray("docvalue_fields")
            .startObject()
            .field("field", VECTOR_FIELD)
            .field("format", "epoch_millis")
            .endObject()
            .endArray()
            .field("_source", false)
            .endObject()
            .toString();

        ResponseException ex = expectThrows(ResponseException.class, () -> searchKNNIndex(TEST_INDEX, query, 1));
        assertTrue(ex.getMessage().contains("does not support custom formats"));
    }

    /**
     * Verifies that when a segment contains documents without the vector field,
     * docvalue_fields gracefully returns no vector for those documents instead of failing.
     * This exercises the EMPTY_DOCVALUE_FETCHER_LEAF path in KNNVectorDVLeafFieldData.
     */
    @SneakyThrows
    public void testDocValueFields_docsWithoutVectorField_returnsEmptyFields() {
        String indexName = TEST_INDEX + "_missing_vector";
        String textField = "title";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject(textField)
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", true).build();

        createKnnIndex(indexName, settings, mapping);

        // Index a doc WITH the vector field
        addKnnDoc(indexName, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        refreshIndex(indexName);

        // Index a doc WITHOUT the vector field (in a separate segment)
        Request request = new Request("POST", "/" + indexName + "/_doc/2?refresh=true");
        request.setJsonEntity("{\"title\": \"no vector here\"}");
        client().performRequest(request);

        // match_all query asking for docvalue_fields on the vector field
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", true)
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();

        Response response = searchKNNIndex(indexName, query, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(2, hits.size());

        // Doc 1 has the vector field
        Map<String, Object> hit1 = hits.get(0);
        assertEquals("1", hit1.get("_id"));
        List<List<Double>> vectorField1 = getDocValueField(hit1, VECTOR_FIELD);
        assertNotNull("Doc with vector should have docvalue_fields", vectorField1);
        assertEquals(DIMENSION, vectorField1.get(0).size());

        // Doc 2 does NOT have the vector field — should have no vector in fields
        Map<String, Object> hit2 = hits.get(1);
        assertEquals("2", hit2.get("_id"));
        List<List<Double>> vectorField2 = getDocValueField(hit2, VECTOR_FIELD);
        assertNull("Doc without vector should not have docvalue_fields for vector", vectorField2);

        deleteKNNIndex(indexName);
    }

    // Nested field tests

    /**
     * Verifies that top-level docvalue_fields on a nested vector path returns null,
     * since nested documents are stored as separate hidden Lucene documents and are
     * not accessible from the parent document's doc values.
     */
    @SneakyThrows
    public void testDocValueFields_nestedField_topLevelReturnsEmpty() {
        String indexName = TEST_INDEX + "_nested_toplevel";
        String nestedField = "paragraphs";
        String nestedVectorField = nestedField + ".embedding";

        createNestedVectorIndex(indexName, nestedField, "embedding", DIMENSION);

        String doc = NestedKnnDocBuilder.create(nestedField)
            .addVectors("embedding", new Float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f })
            .build();
        addKnnDoc(indexName, "1", doc);
        refreshIndex(indexName);

        // Top-level docvalue_fields on a nested path should return nothing for the nested field
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", nestedVectorField)
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(indexName, query, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertEquals(1, hits.size());
        // Nested fields are not accessible via top-level docvalue_fields
        List<List<Double>> vectorField = getDocValueField(hits.get(0), nestedVectorField);
        assertNull(vectorField);

        deleteKNNIndex(indexName);
    }

    /**
     * Verifies that nested vectors can be retrieved via docvalue_fields inside inner_hits
     * of a nested KNN query. Each inner hit should contain the vector in the fields response.
     */
    @SneakyThrows
    public void testDocValueFields_nestedField_innerHitsReturnsVector() {
        String indexName = TEST_INDEX + "_nested_inner";
        String nestedField = "paragraphs";
        String nestedVectorPath = nestedField + ".embedding";

        createNestedVectorIndex(indexName, nestedField, "embedding", DIMENSION);

        String doc1 = NestedKnnDocBuilder.create(nestedField)
            .addVectors("embedding", new Float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new Float[] { 5.0f, 6.0f, 7.0f, 8.0f })
            .build();
        addKnnDoc(indexName, "1", doc1);

        String doc2 = NestedKnnDocBuilder.create(nestedField).addVectors("embedding", new Float[] { 9.0f, 10.0f, 11.0f, 12.0f }).build();
        addKnnDoc(indexName, "2", doc2);
        refreshIndex(indexName);

        // inner_hits with docvalue_fields should return the matched nested vector
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", nestedField)
            .startObject("query")
            .startObject("knn")
            .startObject(nestedVectorPath)
            .field("vector", new float[] { 1.0f, 2.0f, 3.0f, 4.0f })
            .field("k", 3)
            .endObject()
            .endObject()
            .endObject()
            .startObject("inner_hits")
            .array("docvalue_fields", nestedVectorPath)
            .field("_source", false)
            .endObject()
            .endObject()
            .endObject()
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(indexName, query, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        assertFalse(hits.isEmpty());
        for (Map<String, Object> hit : hits) {
            Map<String, Object> innerHits = getInnerHits(hit, nestedField);
            assertNotNull(innerHits);
            List<Map<String, Object>> innerHitsList = getInnerHitsList(innerHits);
            assertFalse(innerHitsList.isEmpty());
            for (Map<String, Object> innerHit : innerHitsList) {
                List<List<Double>> vectorField = getDocValueField(innerHit, nestedVectorPath);
                assertNotNull("inner_hit should contain docvalue_fields vector", vectorField);
                assertEquals(1, vectorField.size());
                assertEquals(DIMENSION, vectorField.get(0).size());
            }
        }

        deleteKNNIndex(indexName);
    }

    /**
     * Verifies that vectors retrieved via inner_hits docvalue_fields exactly match those
     * retrieved via inner_hits _source for the same nested document.
     */
    @SneakyThrows
    public void testDocValueFields_nestedField_innerHitsVectorsMatchSource() {
        String indexName = TEST_INDEX + "_nested_match";
        String nestedField = "paragraphs";
        String nestedVectorPath = nestedField + ".embedding";
        float[] expectedVector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

        createNestedVectorIndex(indexName, nestedField, "embedding", DIMENSION);

        String doc = NestedKnnDocBuilder.create(nestedField).addVectors("embedding", new Float[] { 1.0f, 2.0f, 3.0f, 4.0f }).build();
        addKnnDoc(indexName, "1", doc);
        refreshIndex(indexName);

        // Fetch via inner_hits with _source
        String sourceQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", nestedField)
            .startObject("query")
            .startObject("knn")
            .startObject(nestedVectorPath)
            .field("vector", expectedVector)
            .field("k", 1)
            .endObject()
            .endObject()
            .endObject()
            .startObject("inner_hits")
            .startArray("_source")
            .value(nestedVectorPath)
            .endArray()
            .endObject()
            .endObject()
            .endObject()
            .field("_source", false)
            .endObject()
            .toString();

        Response sourceResponse = searchKNNIndex(indexName, sourceQuery, 1);
        String sourceBody = EntityUtils.toString(sourceResponse.getEntity());
        List<Map<String, Object>> sourceHits = parseSearchHits(sourceBody);
        Map<String, Object> sourceInnerHits = getInnerHits(sourceHits.get(0), nestedField);
        List<Map<String, Object>> sourceInnerList = getInnerHitsList(sourceInnerHits);
        List<Double> sourceVector = getInnerHitSourceVector(sourceInnerList.get(0), "embedding");

        // Fetch via inner_hits with docvalue_fields
        String dvQuery = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", nestedField)
            .startObject("query")
            .startObject("knn")
            .startObject(nestedVectorPath)
            .field("vector", expectedVector)
            .field("k", 1)
            .endObject()
            .endObject()
            .endObject()
            .startObject("inner_hits")
            .array("docvalue_fields", nestedVectorPath)
            .field("_source", false)
            .endObject()
            .endObject()
            .endObject()
            .field("_source", false)
            .endObject()
            .toString();

        Response dvResponse = searchKNNIndex(indexName, dvQuery, 1);
        String dvBody = EntityUtils.toString(dvResponse.getEntity());
        List<Map<String, Object>> dvHits = parseSearchHits(dvBody);
        Map<String, Object> dvInnerHits = getInnerHits(dvHits.get(0), nestedField);
        List<Map<String, Object>> dvInnerList = getInnerHitsList(dvInnerHits);
        List<List<Double>> dvVector = getDocValueField(dvInnerList.get(0), nestedVectorPath);

        // Vectors from _source and docvalue_fields should match
        assertNotNull(dvVector);
        assertEquals(sourceVector.size(), dvVector.get(0).size());
        for (int i = 0; i < sourceVector.size(); i++) {
            assertEquals(sourceVector.get(i), dvVector.get(0).get(i), 0.001);
        }

        deleteKNNIndex(indexName);
    }

    /**
     * Verifies that with expand_nested_docs=true, a nested KNN query returns all matching nested
     * vectors via inner_hits docvalue_fields from a single document with multiple nested objects.
     * Also validates that nested docs without a vector field do not cause errors.
     */
    @SneakyThrows
    public void testDocValueFields_nestedField_expandNestedReturnsAllInnerHitVectors() {
        String indexName = TEST_INDEX + "_nested_multi";
        String nestedField = "paragraphs";
        String nestedVectorPath = nestedField + ".embedding";

        createNestedVectorIndex(indexName, nestedField, "embedding", DIMENSION);

        // Document with 3 nested vectors
        String doc1 = NestedKnnDocBuilder.create(nestedField)
            .addVectors(
                "embedding",
                new Float[] { 1.0f, 2.0f, 3.0f, 4.0f },
                new Float[] { 5.0f, 6.0f, 7.0f, 8.0f },
                new Float[] { 9.0f, 10.0f, 11.0f, 12.0f }
            )
            .build();
        addKnnDoc(indexName, "1", doc1);

        // Document with nested objects where some don't have the vector field
        String doc2 = XContentFactory.jsonBuilder()
            .startObject()
            .startArray(nestedField)
            .startObject()
            .field("embedding", new float[] { 2.0f, 3.0f, 4.0f, 5.0f })
            .endObject()
            .startObject()
            .endObject()
            .startObject()
            .field("embedding", new float[] { 3.0f, 4.0f, 5.0f, 6.0f })
            .endObject()
            .endArray()
            .endObject()
            .toString();
        addKnnDoc(indexName, "2", doc2);
        refreshIndex(indexName);

        // Nested KNN query with expand_nested=true and k=5 to find all nested vectors
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", nestedField)
            .startObject("query")
            .startObject("knn")
            .startObject(nestedVectorPath)
            .field("vector", new float[] { 1.0f, 2.0f, 3.0f, 4.0f })
            .field("k", 5)
            .field("expand_nested_docs", true)
            .endObject()
            .endObject()
            .endObject()
            .startObject("inner_hits")
            .field("size", 10)
            .array("docvalue_fields", nestedVectorPath)
            .field("_source", false)
            .endObject()
            .endObject()
            .endObject()
            .field("_source", false)
            .endObject()
            .toString();

        Response response = searchKNNIndex(indexName, query, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> hits = parseSearchHits(responseBody);

        // Should have 2 parent hits (doc 1 and doc 2)
        assertEquals(2, hits.size());

        int totalInnerHitsWithVector = 0;
        for (Map<String, Object> hit : hits) {
            Map<String, Object> innerHits = getInnerHits(hit, nestedField);
            assertNotNull(innerHits);
            List<Map<String, Object>> innerHitsList = getInnerHitsList(innerHits);
            assertFalse(innerHitsList.isEmpty());
            for (Map<String, Object> innerHit : innerHitsList) {
                List<List<Double>> vectorField = getDocValueField(innerHit, nestedVectorPath);
                if (vectorField != null) {
                    assertEquals(1, vectorField.size());
                    assertEquals(DIMENSION, vectorField.get(0).size());
                    totalInnerHitsWithVector++;
                }
            }
        }

        // Doc 1 has 3 nested vectors, Doc 2 has 2 nested vectors (one nested obj has no vector)
        assertEquals(5, totalInnerHitsWithVector);

        deleteKNNIndex(indexName);
    }

    // Helper methods

    private void createHnswIndex(KNNEngine engine) throws Exception {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.L2.getValue())
            .engine(engine.getName())
            .build();

        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(VECTOR_FIELD)
            .dimension(DIMENSION)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        createKnnIndex(TEST_INDEX, mapping);
    }

    private void createHnswIndex(String indexName, KNNEngine engine) throws Exception {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.L2.getValue())
            .engine(engine.getName())
            .build();

        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(VECTOR_FIELD)
            .dimension(DIMENSION)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        createKnnIndex(indexName, mapping);
    }

    private void indexTestDocuments() throws Exception {
        indexTestDocuments(TEST_INDEX);
    }

    private void indexTestDocuments(String indexName) throws Exception {
        addKnnDoc(indexName, "1", VECTOR_FIELD, Floats.asList(VECTOR_1).toArray());
        addKnnDoc(indexName, "2", VECTOR_FIELD, Floats.asList(VECTOR_2).toArray());
        addKnnDoc(indexName, "3", VECTOR_FIELD, Floats.asList(VECTOR_3).toArray());
        refreshIndex(indexName);
    }

    private String buildDocValueFieldsQuery(float[] queryVector, boolean includeSource) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(VECTOR_FIELD)
            .field("vector", queryVector)
            .field("k", 10)
            .endObject()
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD);
        if (!includeSource) {
            builder.field("_source", false);
        }
        return builder.endObject().toString();
    }

    private String buildSourceQuery(float[] queryVector) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(VECTOR_FIELD)
            .field("vector", queryVector)
            .field("k", 1)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> parseSearchHits(String responseBody) throws IOException {
        Map<String, Object> responseMap = createParser(org.opensearch.common.xcontent.json.JsonXContent.jsonXContent, responseBody).map();
        Map<String, Object> hitsOuter = (Map<String, Object>) responseMap.get("hits");
        return (List<Map<String, Object>>) hitsOuter.get("hits");
    }

    @SuppressWarnings("unchecked")
    private List<List<Double>> getDocValueField(Map<String, Object> hit, String fieldName) {
        Map<String, Object> fields = (Map<String, Object>) hit.get("fields");
        if (fields == null) return null;
        return (List<List<Double>>) fields.get(fieldName);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getFields(Map<String, Object> hit) {
        return (Map<String, Object>) hit.get("fields");
    }

    @SuppressWarnings("unchecked")
    private List<Double> getSourceVector(Map<String, Object> hit, String fieldName) {
        Map<String, Object> source = (Map<String, Object>) hit.get("_source");
        List<Number> raw = (List<Number>) source.get(fieldName);
        List<Double> result = new ArrayList<>();
        for (Number n : raw) {
            result.add(n.doubleValue());
        }
        return result;
    }

    private void createNestedVectorIndex(String indexName, String nestedField, String vectorFieldName, int dimension) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(nestedField)
            .field("type", "nested")
            .startObject("properties")
            .startObject(vectorFieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject("method")
            .field("name", "hnsw")
            .field("space_type", "l2")
            .field("engine", "faiss")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getInnerHits(Map<String, Object> hit, String nestedFieldName) {
        Map<String, Object> innerHitsMap = (Map<String, Object>) hit.get("inner_hits");
        if (innerHitsMap == null) return null;
        return (Map<String, Object>) innerHitsMap.get(nestedFieldName);
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> getInnerHitsList(Map<String, Object> innerHits) {
        Map<String, Object> hitsWrapper = (Map<String, Object>) innerHits.get("hits");
        return (List<Map<String, Object>>) hitsWrapper.get("hits");
    }

    @SuppressWarnings("unchecked")
    private List<Double> getInnerHitSourceVector(Map<String, Object> innerHit, String fieldName) {
        Map<String, Object> source = (Map<String, Object>) innerHit.get("_source");
        List<Number> raw = (List<Number>) source.get(fieldName);
        List<Double> result = new ArrayList<>();
        for (Number n : raw) {
            result.add(n.doubleValue());
        }
        return result;
    }

    private String buildSortedDocValueFieldsQuery() throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .array("docvalue_fields", VECTOR_FIELD)
            .field("_source", false)
            .startObject("sort")
            .field("_id", "asc")
            .endObject()
            .endObject()
            .toString();
    }

    private void assertVectorForDoc(Map<String, List<Double>> docIdToVector, String docId, float[] expected) {
        assertVectorForDoc(docIdToVector, docId, expected, null);
    }

    private void assertVectorForDoc(Map<String, List<Double>> docIdToVector, String docId, float[] expected, String engineName) {
        String prefix = engineName != null ? "[" + engineName + "] " : "";
        List<Double> actual = docIdToVector.get(docId);
        assertNotNull(prefix + "No vector found for doc ID " + docId, actual);
        assertEquals(prefix + "Dimension mismatch for doc " + docId, expected.length, actual.size());
        for (int i = 0; i < expected.length; i++) {
            assertEquals(prefix + "Value mismatch at index " + i + " for doc " + docId, expected[i], actual.get(i), 0.001);
        }
    }
}
