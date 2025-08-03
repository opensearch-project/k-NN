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
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.common.settings.Settings;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;

import static org.opensearch.knn.common.KNNConstants.*;

public class MixedVectorDocumentIT extends KNNRestTestCase {

    private static final String VECTOR_FIELD = "test_vector", METADATA_FIELD = "description", CATEGORY_FIELD = "category";
    private static final int VECTOR_DIMENSION = 128, TOTAL_DOCS = 10, DOCS_WITH_VECTORS = 9;
    private static final KNNEngine[] TEST_ENGINES = { KNNEngine.FAISS, KNNEngine.LUCENE };
    private static final String VECTORIZED = "vectorized", NON_VECTORIZED = "non-vectorized";
    private Set<String> createdIndices = new HashSet<>();

    @Before
    public void setUp() throws Exception {
        super.setUp();
        createdIndices.clear();
    }

    @After
    public void tearDown() throws Exception {
        createdIndices.forEach(this::safeDeleteIndex);
        createdIndices.clear();
        super.tearDown();
    }

    @SneakyThrows
    public void testComprehensiveSearchFunctionalityAcrossAllEngines() {
        for (KNNEngine engine : TEST_ENGINES) {
            String indexName = "test-comprehensive-" + engine.getName().toLowerCase();
            createMixedDocumentIndex(indexName, engine, SpaceType.L2);
            populateStandardMixedDocuments(indexName);

            // Comprehensive validation with compact assertions
            assertKNNSearch(indexName, 5, true, false);
            assertFilteredSearch(indexName, VECTORIZED, NON_VECTORIZED);
            assertDocumentCounts(indexName, 10, 9);
            assertScriptScoring(indexName);
            assertSegmentOperations(indexName);
        }
    }

    @SneakyThrows
    public void testEngineSpecificFunctionality() {
        testFaissFunctionality();
        testLuceneFunctionality();
        testDiskBasedFunctionality();
    }

    private void testFaissFunctionality() throws IOException, ParseException {
        String baseIndex = "test-faiss-";
        createMixedDocumentIndex(baseIndex, KNNEngine.FAISS, SpaceType.L2);
        populateStandardMixedDocuments(baseIndex);

        // Batched segment merging
        String batchedIndex = baseIndex + "-batched";
        createMixedDocumentIndex(batchedIndex, KNNEngine.FAISS, SpaceType.L2);
        IntStream.range(0, 3).forEach(batch -> safeRunBatch(batchedIndex, batch * 5, 3, 2));
        assertSegmentMerge(batchedIndex);

        // Sparse vectors performance test
        String sparseIndex = baseIndex + "-sparse";
        createMixedDocumentIndex(sparseIndex, KNNEngine.FAISS, SpaceType.L2);
        addBatchOfMixedDocuments(sparseIndex, 0, 10, 40);
        assertPerformanceSearch(sparseIndex, 10000);

        // Error handling validation
        assertEquals(200, performKNNSearch(baseIndex, createRandomVector(VECTOR_DIMENSION), 20).getStatusLine().getStatusCode());
        assertEquals(200, performSearch(baseIndex, createTermsAggregationQuery(CATEGORY_FIELD)).getStatusLine().getStatusCode());
    }

    private void testLuceneFunctionality() throws IOException, ParseException {
        String baseIndex = "test-lucene-";

        // Multiple space types validation
        for (SpaceType spaceType : new SpaceType[] { SpaceType.L2, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT }) {
            String indexName = baseIndex + "-" + spaceType.getValue().toLowerCase();
            createMixedDocumentIndex(indexName, KNNEngine.LUCENE, spaceType);
            populateStandardMixedDocuments(indexName);
            assertKNNSearch(indexName, 5, true, false);
        }

        // Native vector handling and query optimization
        String[] testIndices = { baseIndex + "-native", baseIndex + "-optimization" };
        for (String index : testIndices) {
            createMixedDocumentIndex(index, KNNEngine.LUCENE, SpaceType.L2);
            if (index.contains("optimization")) {
                IntStream.rangeClosed(1, 100).forEach(i -> safeAddDocument(index, i, i % 10 != 0, "Document " + i, "test-category"));
                assertPerformanceSearch(index, 5000);
            } else {
                populateStandardMixedDocuments(index);
                assertVectorDocumentCount(index, 9);
            }
        }
    }

    private void testDiskBasedFunctionality() throws IOException, ParseException {
        String baseIndex = "test-disk-";
        String[] indices = { "-basic", "-large", "-memory", "-optimization" };

        for (String suffix : indices) {
            String indexName = baseIndex + suffix;
            createDiskBasedMixedDocumentIndex(indexName, KNNEngine.FAISS, SpaceType.L2);

            switch (suffix) {
                case "-basic":
                    populateStandardMixedDocuments(indexName);
                    updateIndexSettings(indexName, "\"index.knn.advanced.approximate_threshold\": 0");
                    refreshAllIndices();
                    assertKNNSearch(indexName, 5, true, false);
                    break;
                case "-large":
                    addBatchOfMixedDocuments(indexName, 0, 90, 10);
                    assertSegmentMerge(indexName);
                    break;
                case "-memory":
                    populateStandardMixedDocuments(indexName);
                    assertEquals(200, performKNNSearch(indexName, createRandomVector(VECTOR_DIMENSION), 3).getStatusLine().getStatusCode());
                    break;
                case "-optimization":
                    IntStream.range(0, 5).forEach(batch -> safeRunBatch(indexName, batch * 10, 8, 2));
                    assertEquals(200, performRequest("POST", "/" + indexName + "/_forcemerge").getStatusLine().getStatusCode());
                    assertKNNSearch(indexName, 5, true, false);
                    break;
            }
        }
    }

    // Utility methods - significantly reduced
    private void safeDeleteIndex(String indexName) {
        try {
            if (checkIndexExists(indexName)) deleteKNNIndex(indexName);
        } catch (IOException e) {}
    }

    private void safeRunBatch(String indexName, int startId, int vectorDocs, int nonVectorDocs) {
        try {
            addBatchOfMixedDocuments(indexName, startId, vectorDocs, nonVectorDocs);
            refreshAllIndices();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void safeAddDocument(String indexName, int docId, boolean hasVector, String description, String category) {
        try {
            if (hasVector) addDocumentWithVector(indexName, docId, description, category);
            else addDocumentWithoutVector(indexName, docId, description, category);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void createMixedDocumentIndex(String indexName, KNNEngine engine, SpaceType spaceType) throws IOException {
        createKnnIndex(indexName, createIndexSettings(engine, spaceType, false), createIndexMapping(engine, spaceType, false));
        createdIndices.add(indexName);
    }

    private void createDiskBasedMixedDocumentIndex(String indexName, KNNEngine engine, SpaceType spaceType) throws IOException {
        createKnnIndex(indexName, createIndexSettings(engine, spaceType, true), createIndexMapping(engine, spaceType, true));
        createdIndices.add(indexName);
    }

    private Settings createIndexSettings(KNNEngine engine, SpaceType spaceType, boolean diskBased) {
        Settings.Builder builder = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn.remote_index_build.enabled", false); // Disable remote index building
        if (diskBased) builder.put("index.knn.advanced.approximate_threshold", 0);
        return builder.build();
    }

    private String createIndexMapping(KNNEngine engine, SpaceType spaceType, boolean diskBased) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "knn_vector")
            .field(DIMENSION, VECTOR_DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, engine.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field("ef_construction", diskBased ? 128 : 256)
            .field("m", diskBased ? 16 : 8)
            .endObject()
            .endObject()
            .endObject()
            .startObject(METADATA_FIELD)
            .field("type", "text")
            .endObject()
            .startObject(CATEGORY_FIELD)
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    private void populateStandardMixedDocuments(String indexName) throws IOException {
        IntStream.rangeClosed(1, DOCS_WITH_VECTORS)
            .forEach(i -> safeAddDocument(indexName, i, true, "Document " + i + " with vector", VECTORIZED));
        addDocumentWithoutVector(indexName, TOTAL_DOCS, "Document 10 without vector", NON_VECTORIZED);
        refreshAllIndices();
    }

    private void addDocumentWithVector(String indexName, int docId, String description, String category) throws IOException {
        addDocument(
            indexName,
            docId,
            XContentFactory.jsonBuilder()
                .startObject()
                .field(VECTOR_FIELD, createRandomVector(VECTOR_DIMENSION))
                .field(METADATA_FIELD, description)
                .field(CATEGORY_FIELD, category)
                .endObject()
        );
    }

    private void addDocumentWithoutVector(String indexName, int docId, String description, String category) throws IOException {
        addDocument(
            indexName,
            docId,
            XContentFactory.jsonBuilder().startObject().field(METADATA_FIELD, description).field(CATEGORY_FIELD, category).endObject()
        );
    }

    private void addDocument(String indexName, int docId, XContentBuilder doc) throws IOException {
        Request request = new Request("POST", "/" + indexName + "/_doc/" + docId + "?refresh=true");
        request.setJsonEntity(doc.toString());
        client().performRequest(request);
    }

    private void addBatchOfMixedDocuments(String indexName, int startId, int vectorDocs, int nonVectorDocs) throws IOException {
        int currentId = startId;
        for (int i = 0; i < vectorDocs; i++)
            addDocumentWithVector(indexName, ++currentId, "Batch document " + currentId + " with vector", "batch-category");
        for (int i = 0; i < nonVectorDocs; i++)
            addDocumentWithoutVector(indexName, ++currentId, "Batch document " + currentId + " without vector", "batch-category");
    }

    // Compact assertion methods
    private void assertKNNSearch(String indexName, int k, boolean shouldHaveHits, boolean shouldHaveDoc10) throws IOException,
        ParseException {
        String response = performKNNSearchAndGetBody(indexName, createRandomVector(VECTOR_DIMENSION), k);
        assertEquals(shouldHaveHits, response.contains("\"hits\":"));
        assertEquals(shouldHaveDoc10, response.contains("\"10\""));
    }

    private void assertFilteredSearch(String indexName, String... categories) throws IOException, ParseException {
        String response = performSearchAndGetBody(indexName, createBoolQuery(categories));
        for (String category : categories)
            assertTrue(response.contains(category));
    }

    private void assertDocumentCounts(String indexName, int totalCount, int vectorCount) throws IOException, ParseException {
        assertTrue(performCountAndGetBody(indexName).contains("\"count\":" + totalCount));
        assertTrue(performSearchCountAndGetBody(indexName, createExistsQuery(VECTOR_FIELD)).contains("\"count\":" + vectorCount));
    }

    private void assertScriptScoring(String indexName) throws IOException, ParseException {
        assertTrue(performSearchAndGetBody(indexName, createScriptScoreQuery(createRandomVector(VECTOR_DIMENSION))).contains("\"hits\":"));
    }

    private void assertSegmentOperations(String indexName) throws IOException {
        assertEquals(200, performRequest("POST", "/" + indexName + "/_forcemerge?max_num_segments=1").getStatusLine().getStatusCode());
        refreshAllIndices();
        assertEquals(200, performKNNSearch(indexName, createRandomVector(VECTOR_DIMENSION), 3).getStatusLine().getStatusCode());
    }

    private void assertSegmentMerge(String indexName) throws IOException, ParseException {
        assertEquals(200, performRequest("POST", "/" + indexName + "/_forcemerge?max_num_segments=1").getStatusLine().getStatusCode());
        assertTrue(performKNNSearchAndGetBody(indexName, createRandomVector(VECTOR_DIMENSION), 5).contains("\"hits\":"));
    }

    private void assertPerformanceSearch(String indexName, long maxTime) throws IOException, ParseException {
        long start = System.currentTimeMillis();
        String response = performKNNSearchAndGetBody(indexName, createRandomVector(VECTOR_DIMENSION), 5);
        assertTrue("Search time should be reasonable", System.currentTimeMillis() - start < maxTime);
        assertTrue(response.contains("\"hits\":"));
    }

    private void assertVectorDocumentCount(String indexName, int expectedCount) throws IOException, ParseException {
        assertTrue(performSearchAndGetBody(indexName, createExistsQuery(VECTOR_FIELD)).contains("\"total\":{\"value\":" + expectedCount));
    }

    // Compact query builders
    private XContentBuilder createBoolQuery(String... categoryValues) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query").startObject("bool").startArray("should");
        for (String value : categoryValues)
            builder.startObject().startObject("term").field(CATEGORY_FIELD, value).endObject().endObject();
        return builder.endArray().endObject().endObject().endObject();
    }

    private XContentBuilder createExistsQuery(String field) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("exists")
            .field("field", field)
            .endObject()
            .endObject()
            .endObject();
    }

    private XContentBuilder createScriptScoreQuery(float[] queryVector) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("script_score")
            .startObject("query")
            .startObject("match_all")
            .endObject()
            .endObject()
            .startObject("script")
            .field(
                "source",
                "if (doc['"
                    + VECTOR_FIELD
                    + "'].size() > 0) { cosineSimilarity(params.query_vector, doc['"
                    + VECTOR_FIELD
                    + "']) + 1.0 } else { 0.1 }"
            )
            .startObject("params")
            .field("query_vector", queryVector)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
    }

    private XContentBuilder createTermsAggregationQuery(String field) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("aggs")
            .startObject("category_terms")
            .startObject("terms")
            .field("field", field)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
    }

    // Compact search helpers
    private String performKNNSearchAndGetBody(String indexName, float[] queryVector, int k) throws IOException, ParseException {
        return EntityUtils.toString(performKNNSearch(indexName, queryVector, k).getEntity());
    }

    private String performSearchAndGetBody(String indexName, XContentBuilder query) throws IOException, ParseException {
        return EntityUtils.toString(performSearch(indexName, query).getEntity());
    }

    private String performCountAndGetBody(String indexName) throws IOException, ParseException {
        return EntityUtils.toString(performRequest("GET", "/" + indexName + "/_count").getEntity());
    }

    private String performSearchCountAndGetBody(String indexName, XContentBuilder query) throws IOException, ParseException {
        Request request = new Request("POST", "/" + indexName + "/_count");
        request.setJsonEntity(query.toString());
        return EntityUtils.toString(client().performRequest(request).getEntity());
    }

    private Response performKNNSearch(String indexName, float[] queryVector, int k) throws IOException {
        XContentBuilder query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(VECTOR_FIELD)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return performSearch(indexName, query);
    }

    private Response performSearch(String indexName, XContentBuilder query) throws IOException {
        Request request = new Request("POST", "/" + indexName + "/_search");
        request.setJsonEntity(query.toString());
        return client().performRequest(request);
    }

    private Response performRequest(String method, String endpoint) throws IOException {
        return client().performRequest(new Request(method, endpoint));
    }

    private void updateIndexSettings(String indexName, String settings) throws IOException {
        Request request = new Request("PUT", "/" + indexName + "/_settings");
        request.setJsonEntity("{" + settings + "}");
        client().performRequest(request);
    }

    private float[] createRandomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++)
            vector[i] = (float) (Math.random() * 10.0);
        return vector;
    }

    private boolean checkIndexExists(String indexName) throws IOException {
        try {
            return client().performRequest(new Request("HEAD", "/" + indexName)).getStatusLine().getStatusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }
}
