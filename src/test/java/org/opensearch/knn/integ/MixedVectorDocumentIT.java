/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
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

/**
 * Integration tests for mixed vector documents - indices containing both documents with vectors
 * and documents without vectors across different k-NN engines.
 */
public class MixedVectorDocumentIT extends KNNRestTestCase {

    private static final String VECTOR_FIELD = "test_vector";
    private static final String METADATA_FIELD = "description";
    private static final String CATEGORY_FIELD = "category";
    private static final int VECTOR_DIMENSION = 128;
    private static final int TOTAL_DOCS = 10;
    private static final int DOCS_WITH_VECTORS = 9;
    private static final KNNEngine[] TEST_ENGINES = { KNNEngine.FAISS, KNNEngine.LUCENE };
    private static final String VECTORIZED = "vectorized";
    private static final String NON_VECTORIZED = "non-vectorized";

    // Test constants following established patterns
    private static final String HITS_PATTERN = "\"hits\":";
    private static final String COUNT_PATTERN = "\"count\":";
    private static final String TOTAL_PATTERN = "\"total\":{\"value\":";

    private final Set<String> createdIndices = new HashSet<>();

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

    /**
     * Test comprehensive search functionality across all supported engines
     */
    @SneakyThrows
    public void testKNNIndex_mixedVectorDocuments_comprehensiveSearchFunctionality() {
        for (KNNEngine engine : TEST_ENGINES) {
            String indexName = "test-comprehensive-" + engine.getName().toLowerCase() + "-" + randomAlphaOfLength(5).toLowerCase();
            createMixedDocumentIndex(indexName, engine, SpaceType.L2);
            populateStandardMixedDocuments(indexName);

            // Validate search functionality with proper result verification
            assertKNNSearchReturnsValidResults(indexName, 5);
            assertFilteredSearchReturnsCategories(indexName, VECTORIZED, NON_VECTORIZED);
            assertDocumentCounts(indexName);
            assertScriptScoringWorks(indexName);
            assertSegmentOperationsSucceed(indexName);

            logger.info("Completed comprehensive search functionality test for engine: {}", engine.getName());
        }
    }

    /**
     * Test engine-specific functionality
     */
    @SneakyThrows
    public void testKNNIndex_engineSpecificFunctionality() {
        testFaissFunctionality();
        testLuceneFunctionality();
        testDiskBasedFunctionality();
    }

    /**
     * Test FAISS-specific features including batched operations and performance
     */
    private void testFaissFunctionality() throws Exception {
        String baseIndex = "test-faiss-base-" + randomAlphaOfLength(5).toLowerCase();
        createMixedDocumentIndex(baseIndex, KNNEngine.FAISS, SpaceType.L2);
        populateStandardMixedDocuments(baseIndex);

        // Batched segment merging with unique IDs
        String batchedIndex = "test-faiss-batched-" + randomAlphaOfLength(5).toLowerCase();
        createMixedDocumentIndex(batchedIndex, KNNEngine.FAISS, SpaceType.L2);
        IntStream.range(0, 3).forEach(batch -> safeRunBatch(batchedIndex, 1000 + batch * 10, 3));
        assertSegmentMergeSucceeds(batchedIndex);

        // Large dataset test with unique IDs
        String largeIndex = "test-faiss-large-" + randomAlphaOfLength(5).toLowerCase();
        createMixedDocumentIndex(largeIndex, KNNEngine.FAISS, SpaceType.L2);
        addBatchOfMixedDocuments(largeIndex, 2000, 10, 40);
        assertSearchPerformanceReasonable(largeIndex);

        // Error handling validation
        assertEquals(200, performKNNSearch(baseIndex, createRandomVector(VECTOR_DIMENSION), 20).getStatusLine().getStatusCode());
        assertEquals(200, performSearch(baseIndex, createTermsAggregationQuery(CATEGORY_FIELD).toString()).getStatusLine().getStatusCode());

        logger.info("Completed FAISS functionality tests");
    }

    /**
     * Test Lucene-specific features with compatible space types
     */
    private void testLuceneFunctionality() throws Exception {
        String baseIndex = "test-lucene-base-" + randomAlphaOfLength(5).toLowerCase();

        // Test compatible space types for Lucene engine (excluding INNER_PRODUCT for compatibility)
        for (SpaceType spaceType : new SpaceType[] { SpaceType.L2, SpaceType.COSINESIMIL }) {
            String indexName = baseIndex + "-" + spaceType.getValue().toLowerCase() + "-" + randomAlphaOfLength(3).toLowerCase();
            createMixedDocumentIndex(indexName, KNNEngine.LUCENE, spaceType);
            populateStandardMixedDocuments(indexName);
            assertKNNSearchReturnsValidResults(indexName, 5);
        }

        // Native vector handling
        String nativeIndex = baseIndex + "-native-" + randomAlphaOfLength(5).toLowerCase();
        createMixedDocumentIndex(nativeIndex, KNNEngine.LUCENE, SpaceType.L2);
        populateStandardMixedDocuments(nativeIndex);
        assertVectorDocumentCount(nativeIndex);

        // Query optimization test with unique IDs
        String optimizationIndex = baseIndex + "-optimization-" + randomAlphaOfLength(5).toLowerCase();
        createMixedDocumentIndex(optimizationIndex, KNNEngine.LUCENE, SpaceType.L2);
        IntStream.rangeClosed(3000, 3050)
            .forEach(i -> safeAddDocument(optimizationIndex, i, i % 10 != 0, "Document " + i, "test-category"));
        assertSearchPerformanceReasonable(optimizationIndex);

        logger.info("Completed Lucene functionality tests");
    }

    /**
     * Test disk-based functionality with various configurations
     */
    private void testDiskBasedFunctionality() throws Exception {
        String baseIndex = "test-disk-base-" + randomAlphaOfLength(5).toLowerCase().toLowerCase();

        // Basic disk-based functionality
        String basicIndex = baseIndex + "-basic";
        createDiskBasedMixedDocumentIndex(basicIndex, KNNEngine.FAISS, SpaceType.L2);
        populateStandardMixedDocuments(basicIndex);
        updateIndexSettings(basicIndex);
        refreshAllNonSystemIndices();
        assertKNNSearchReturnsValidResults(basicIndex, 5);

        // Large dataset with disk-based storage using unique IDs
        String largeIndex = baseIndex + "-large";
        createDiskBasedMixedDocumentIndex(largeIndex, KNNEngine.FAISS, SpaceType.L2);
        addBatchOfMixedDocuments(largeIndex, 4000, 90, 10);
        assertSegmentMergeSucceeds(largeIndex);

        // Memory optimization test
        String memoryIndex = baseIndex + "-memory";
        createDiskBasedMixedDocumentIndex(memoryIndex, KNNEngine.FAISS, SpaceType.L2);
        populateStandardMixedDocuments(memoryIndex);
        assertEquals(200, performKNNSearch(memoryIndex, createRandomVector(VECTOR_DIMENSION), 3).getStatusLine().getStatusCode());

        // Optimization with batched operations using unique IDs
        String optimizationIndex = baseIndex + "-optimization";
        createDiskBasedMixedDocumentIndex(optimizationIndex, KNNEngine.FAISS, SpaceType.L2);
        IntStream.range(0, 5).forEach(batch -> safeRunBatch(optimizationIndex, 5000 + batch * 10, 8));
        assertEquals(
            200,
            client().performRequest(new Request("POST", "/" + optimizationIndex + "/_forcemerge")).getStatusLine().getStatusCode()
        );
        assertKNNSearchReturnsValidResults(optimizationIndex, 5);

        logger.info("Completed disk-based functionality tests");
    }

    // Utility methods with proper error handling
    private void safeDeleteIndex(String indexName) {
        try {
            if (checkIndexExists(indexName)) {
                deleteKNNIndex(indexName);
                logger.debug("Successfully deleted index: {}", indexName);
            }
        } catch (IOException e) {
            logger.warn("Failed to delete index {}: {}", indexName, e.getMessage());
        }
    }

    private void safeRunBatch(String indexName, int startId, int vectorDocs) {
        try {
            addBatchOfMixedDocuments(indexName, startId, vectorDocs, 2);
            refreshIndex(indexName); // Use individual refresh without sleep
        } catch (IOException e) {
            fail("Failed to run batch for index " + indexName + ": " + e.getMessage());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void safeAddDocument(String indexName, int docId, boolean hasVector, String description, String category) {
        try {
            if (hasVector) {
                addDocumentWithVector(indexName, docId, description, category);
            } else {
                addDocumentWithoutVector(indexName, docId, description, category);
            }
        } catch (IOException e) {
            fail("Failed to add document " + docId + " to index " + indexName + ": " + e.getMessage());
        }
    }

    private void createMixedDocumentIndex(String indexName, KNNEngine engine, SpaceType spaceType) throws IOException {
        createKnnIndex(indexName, createIndexSettings(engine, spaceType, false), createIndexMapping(engine, spaceType, false));
        createdIndices.add(indexName);
        logger.debug(
            "Created mixed document index: {} with engine: {} and space type: {}",
            indexName,
            engine.getName(),
            spaceType.getValue()
        );
    }

    private void createDiskBasedMixedDocumentIndex(String indexName, KNNEngine engine, SpaceType spaceType) throws IOException {
        createKnnIndex(indexName, createIndexSettings(engine, spaceType, true), createIndexMapping(engine, spaceType, true));
        createdIndices.add(indexName);
        logger.debug(
            "Created disk-based mixed document index: {} with engine: {} and space type: {}",
            indexName,
            engine.getName(),
            spaceType.getValue()
        );
    }

    private Settings createIndexSettings(KNNEngine engine, SpaceType spaceType, boolean diskBased) {
        Settings.Builder builder = Settings.builder().put("index.knn", true).put("number_of_shards", 1).put("number_of_replicas", 0);

        if (diskBased) {
            builder.put("index.knn.advanced.approximate_threshold", 0);
        }
        return builder.build();
    }

    private String createIndexMapping(KNNEngine engine, SpaceType spaceType, boolean diskBased) throws IOException {
        // Use engine-appropriate parameters based on existing test patterns
        int efConstruction = getOptimalEfConstruction(engine, diskBased);
        int m = getOptimalM(engine, diskBased);

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
            .field("ef_construction", efConstruction)
            .field("m", m)
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

    private int getOptimalEfConstruction(KNNEngine engine, boolean diskBased) {
        if (engine == KNNEngine.FAISS) {
            return diskBased ? 128 : 512;
        } else if (engine == KNNEngine.LUCENE) {
            return diskBased ? 100 : 256;
        }
        return 256; // default
    }

    private int getOptimalM(KNNEngine engine, boolean diskBased) {
        if (engine == KNNEngine.FAISS) {
            return 16; // Consistent for FAISS
        } else if (engine == KNNEngine.LUCENE) {
            return diskBased ? 8 : 16;
        }
        return 16; // default
    }

    private void populateStandardMixedDocuments(String indexName) throws Exception {
        IntStream.rangeClosed(1, DOCS_WITH_VECTORS)
            .forEach(i -> safeAddDocument(indexName, i, true, "Document " + i + " with vector", VECTORIZED));
        addDocumentWithoutVector(indexName, TOTAL_DOCS, "Document 10 without vector", NON_VECTORIZED);
        refreshIndex(indexName);
        logger.debug(
            "Populated {} documents in index: {} ({} with vectors, {} without)",
            TOTAL_DOCS,
            indexName,
            DOCS_WITH_VECTORS,
            TOTAL_DOCS - DOCS_WITH_VECTORS
        );
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

    private void addBatchOfMixedDocuments(String indexName, int startId, int vectorDocs, int nonVectorDocs) throws Exception {
        int currentId = startId;
        for (int i = 0; i < vectorDocs; i++) {
            addDocumentWithVector(indexName, currentId++, "Batch document " + currentId + " with vector", "batch-category");
        }
        for (int i = 0; i < nonVectorDocs; i++) {
            addDocumentWithoutVector(indexName, currentId++, "Batch document " + currentId + " without vector", "batch-category");
        }
        refreshIndex(indexName); // Use individual refresh without sleep
    }

    // Enhanced assertion methods with proper result validation
    private void assertKNNSearchReturnsValidResults(String indexName, int k) throws IOException, ParseException {
        String response = performKNNSearchAndGetBody(indexName, VECTOR_FIELD, createRandomVector(VECTOR_DIMENSION), k);
        assertTrue("Should return search results for " + indexName, response.contains(HITS_PATTERN));

        // Verify non-vector document (doc 10) is not in k-NN results
        assertFalse("Non-vector document should not appear in k-NN results for " + indexName, response.contains("\"_id\":\"10\""));

        // Validate result structure
        assertTrue("Response should contain total hits for " + indexName, response.contains("\"total\":"));
        logger.debug("k-NN search validation passed for index: {}", indexName);
    }

    private void assertFilteredSearchReturnsCategories(String indexName, String... categories) throws IOException, ParseException {
        String response = performSearchAndGetBody(indexName, createBoolTermQuery(CATEGORY_FIELD, categories));
        for (String category : categories) {
            assertTrue("Filtered search should return results for category " + category + " in " + indexName, response.contains(category));
        }
        logger.debug("Filtered search validation passed for index: {}", indexName);
    }

    private void assertDocumentCounts(String indexName) throws IOException, ParseException {
        String countResponse = performCountAndGetBody(indexName);
        assertTrue(
            "Should have " + MixedVectorDocumentIT.TOTAL_DOCS + " total documents for " + indexName,
            countResponse.contains(COUNT_PATTERN + MixedVectorDocumentIT.TOTAL_DOCS)
        );

        String vectorCountResponse = performSearchCountAndGetBody(indexName, createExistsQuery(VECTOR_FIELD));
        assertTrue(
            "Should have " + MixedVectorDocumentIT.DOCS_WITH_VECTORS + " documents with vectors for " + indexName,
            vectorCountResponse.contains(COUNT_PATTERN + MixedVectorDocumentIT.DOCS_WITH_VECTORS)
        );

        logger.debug(
            "Document count validation passed for index: {} (total: {}, with vectors: {})",
            indexName,
            MixedVectorDocumentIT.TOTAL_DOCS,
            MixedVectorDocumentIT.DOCS_WITH_VECTORS
        );
    }

    private void assertScriptScoringWorks(String indexName) throws IOException, ParseException {
        String response = performSearchAndGetBody(
            indexName,
            createVectorScriptScoreQuery(VECTOR_FIELD, createRandomVector(VECTOR_DIMENSION))
        );
        assertTrue("Script scoring should return results for " + indexName, response.contains(HITS_PATTERN));
        assertTrue("Script scoring should return scored results for " + indexName, response.contains("\"_score\":"));
        logger.debug("Script scoring validation passed for index: {}", indexName);
    }

    private void assertSegmentOperationsSucceed(String indexName) throws Exception {
        assertEquals(
            "Force merge should succeed for " + indexName,
            200,
            client().performRequest(new Request("POST", "/" + indexName + "/_forcemerge?max_num_segments=1"))
                .getStatusLine()
                .getStatusCode()
        );
        refreshAllNonSystemIndices();
        assertEquals(
            "k-NN search after segment operations should succeed for " + indexName,
            200,
            performKNNSearch(indexName, createRandomVector(VECTOR_DIMENSION), 3).getStatusLine().getStatusCode()
        );
        logger.debug("Segment operations validation passed for index: {}", indexName);
    }

    private void assertSegmentMergeSucceeds(String indexName) throws IOException, ParseException {
        assertEquals(
            "Force merge should succeed for " + indexName,
            200,
            client().performRequest(new Request("POST", "/" + indexName + "/_forcemerge?max_num_segments=1"))
                .getStatusLine()
                .getStatusCode()
        );
        String response = performKNNSearchAndGetBody(indexName, VECTOR_FIELD, createRandomVector(VECTOR_DIMENSION), 5);
        assertTrue("Search after segment merge should return results for " + indexName, response.contains(HITS_PATTERN));
        logger.debug("Segment merge validation passed for index: {}", indexName);
    }

    private void assertSearchPerformanceReasonable(String indexName) throws IOException, ParseException {
        long start = System.currentTimeMillis();
        String response = performKNNSearchAndGetBody(indexName, VECTOR_FIELD, createRandomVector(VECTOR_DIMENSION), 5);
        long duration = System.currentTimeMillis() - start;

        // Use lenient timeout - 30 seconds should be reasonable for any test environment
        assertTrue("Search should complete within reasonable time for " + indexName + " (took " + duration + "ms)", duration < 30000);
        assertTrue("Performance search should return results for " + indexName, response.contains(HITS_PATTERN));
        logger.debug("Performance search validation passed for index: {} (duration: {}ms)", indexName, duration);
    }

    private void assertVectorDocumentCount(String indexName) throws IOException, ParseException {
        String response = performSearchAndGetBody(indexName, createExistsQuery(VECTOR_FIELD));
        assertTrue(
            "Should have " + MixedVectorDocumentIT.DOCS_WITH_VECTORS + " documents with vectors for " + indexName,
            response.contains(TOTAL_PATTERN + MixedVectorDocumentIT.DOCS_WITH_VECTORS)
        );
        logger.debug(
            "Vector document count validation passed for index: {} (expected: {})",
            indexName,
            MixedVectorDocumentIT.DOCS_WITH_VECTORS
        );
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
        return performSearch(indexName, query.toString());
    }

    private void updateIndexSettings(String indexName) throws IOException {
        Request request = new Request("PUT", "/" + indexName + "/_settings");
        request.setJsonEntity("{" + "\"index.knn.advanced.approximate_threshold\": 0" + "}");
        client().performRequest(request);
    }
}
