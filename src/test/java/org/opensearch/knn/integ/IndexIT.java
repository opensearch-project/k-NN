/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * This class contains integration tests for index
 */
@Log4j2
public class IndexIT extends KNNRestTestCase {
    private static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (IndexIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of IndexIT Class is null");
        }
        URL testIndexVectors = IndexIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = IndexIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        URL groundTruthValues = IndexIT.class.getClassLoader().getResource("data/test_ground_truth_l2_100.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        assert groundTruthValues != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath(), groundTruthValues.getPath());
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testFaissHnsw_when1000Data_thenRecallIsAboveNinePointZero() {
        // Create Index
        createKnnHnswIndex(KNNEngine.FAISS, INDEX_NAME, FIELD_NAME, 128);
        ingestTestData(INDEX_NAME, FIELD_NAME);

        int k = 100;
        for (int i = 0; i < testData.queries.length; i++) {
            List<KNNResult> knnResults = runKnnQuery(INDEX_NAME, FIELD_NAME, testData.queries[i], k);
            float recall = getRecall(
                Set.of(Arrays.copyOf(testData.groundTruthValues[i], k)),
                knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet())
            );
            assertTrue("Recall: " + recall, recall > 0.9);
        }
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Deletes a vector doc, creating a new segment with deleted docs but no docs present.
     * Validates k-NN search functionality works without errors.
     */
    @SneakyThrows
    public void testIndexWithNonVectorFields_whenValid_thenSucceed() {
        String indexName = INDEX_NAME + "_mixed";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 128)
            .field("data_type", VectorDataType.FLOAT.getValue())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject("category")
            .field("type", "keyword")
            .endObject()
            .startObject("description")
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        // Add mixed content
        for (int i = 0; i < 10; i++) {
            addKnnDoc(indexName, String.valueOf(i), FIELD_NAME, Floats.asList(testData.indexData.vectors[i]).toArray());
        }
        addNonKNNDoc(indexName, "10", "description", "Test document");
        assertEquals(11, getDocCount(indexName));
        int segmentCountBeforeDelete = getTotalSegmentCount(indexName);
        deleteKnnDoc(indexName, "0");
        assertEquals(10, getDocCount(indexName));

        flush(indexName, true);
        int segmentCountAfterDelete = getTotalSegmentCount(indexName);
        assertTrue(segmentCountAfterDelete > segmentCountBeforeDelete);

        // Test search
        List<KNNResult> results = runKnnQuery(indexName, FIELD_NAME, testData.queries[0], 5);
        assertTrue(results.size() > 0);
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Deletes a vector doc, creating a new segment with deleted docs but no docs present.
     * Validates k-NN search functionality works without errors for Lucene engine.
     */
    @SneakyThrows
    public void testLuceneWithNonVectorFields_whenValid_thenSucceed() {
        String indexName = INDEX_NAME + "_lucene_mixed";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 128)
            .field("data_type", VectorDataType.FLOAT.getValue())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .startObject("category")
            .field("type", "keyword")
            .endObject()
            .startObject("description")
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        // Add mixed content
        for (int i = 0; i < 10; i++) {
            addKnnDoc(indexName, String.valueOf(i), FIELD_NAME, Floats.asList(testData.indexData.vectors[i]).toArray());
        }
        addNonKNNDoc(indexName, "10", "description", "Test document");
        assertEquals(11, getDocCount(indexName));
        int segmentCountBeforeDelete = getTotalSegmentCount(indexName);
        deleteKnnDoc(indexName, "0");
        assertEquals(10, getDocCount(indexName));

        flush(indexName, true);
        int segmentCountAfterDelete = getTotalSegmentCount(indexName);
        assertTrue(segmentCountAfterDelete > segmentCountBeforeDelete);

        // Test search
        List<KNNResult> results = runKnnQuery(indexName, FIELD_NAME, testData.queries[0], 5);
        assertTrue(results.size() > 0);
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Creates separate segments: one with vector docs, one with only non-vector doc.
     * Validates k-NN search functionality works without errors.
     */
    @SneakyThrows
    public void testMixedSegmentsWithFaiss_whenValid_thenSucceed() {
        String indexName = INDEX_NAME + "_faiss_mixed_segments";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 128)
            .field("data_type", VectorDataType.FLOAT.getValue())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject("category")
            .field("type", "keyword")
            .endObject()
            .startObject("description")
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        // Add vector docs
        for (int i = 0; i < 11; i++) {
            addKnnDoc(indexName, String.valueOf(i), FIELD_NAME, Floats.asList(testData.indexData.vectors[i]).toArray());
        }
        flush(indexName, true);

        // Add non-vector doc (gets its own segment)
        addNonKNNDoc(indexName, "11", "description", "Test document");
        flush(indexName, true);

        flush(indexName, true);
        int segmentCount = getTotalSegmentCount(indexName);
        assertTrue(segmentCount >= 2);

        // Test search on mixed segments
        List<KNNResult> results = runKnnQuery(indexName, FIELD_NAME, testData.queries[0], 5);
        assertTrue(results.size() > 0);
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Creates separate segments: one with vector docs, one with only non-vector doc.
     * Validates k-NN search functionality works without errors for Lucene engine.
     */
    @SneakyThrows
    public void testMixedSegmentsWithLucene_whenValid_thenSucceed() {
        String indexName = INDEX_NAME + "_lucene_mixed_segments";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 128)
            .field("data_type", VectorDataType.FLOAT.getValue())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .startObject("category")
            .field("type", "keyword")
            .endObject()
            .startObject("description")
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        // Add vector docs
        for (int i = 0; i < 11; i++) {
            addKnnDoc(indexName, String.valueOf(i), FIELD_NAME, Floats.asList(testData.indexData.vectors[i]).toArray());
        }
        flush(indexName, true);

        // Add non-vector doc (gets its own segment)
        addNonKNNDoc(indexName, "11", "description", "Test document");
        flush(indexName, true);

        flush(indexName, true);
        int segmentCount = getTotalSegmentCount(indexName);
        assertTrue(segmentCount >= 2);

        // Test search on mixed segments
        List<KNNResult> results = runKnnQuery(indexName, FIELD_NAME, testData.queries[0], 5);
        assertTrue(results.size() > 0);
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Creates a doc with vector field, then updates it to remove the vector field.
     * Validates k-NN search functionality works without errors.
     */
    @SneakyThrows
    public void testVectorFieldRemovalByUpdate_whenValid_thenSucceed() {
        String indexName = INDEX_NAME + "_vector_removal";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 128)
            .field("data_type", VectorDataType.FLOAT.getValue())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject("description")
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        // Add doc with both vector and text field
        String docId = "0";
        String docWithBoth = XContentFactory.jsonBuilder()
            .startObject()
            .field(FIELD_NAME, Floats.asList(testData.indexData.vectors[0]).toArray())
            .field("description", "Test document")
            .endObject()
            .toString();
        addKnnDoc(indexName, docId, docWithBoth);

        // Update doc to remove vector field, keeping only text field
        addNonKNNDoc(indexName, docId, "description", "Updated test document");

        flush(indexName, true);

        // Verify index has one doc with no vector field and search returns no results
        assertEquals(1, getDocCount(indexName));
        List<KNNResult> results = runKnnQuery(indexName, FIELD_NAME, testData.queries[0], 5);
        assertEquals(0, results.size());
    }

    /**
     * Test segment with knn_vector field mapping but no docs containing the vector field.
     * Creates a doc with vector field, then updates it to remove the vector field.
     * Validates k-NN search functionality works without errors for Lucene engine.
     */
    @SneakyThrows
    public void testVectorFieldRemovalByUpdateLucene_whenValid_thenSucceed() {
        String indexName = INDEX_NAME + "_vector_removal_lucene";

        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 128)
            .field("data_type", VectorDataType.FLOAT.getValue())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("space_type", SpaceType.L2.getValue())
            .field("engine", KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .startObject("description")
            .field("type", "text")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, mapping);

        // Add doc with both vector and text field
        String docId = "0";
        String docWithBoth = XContentFactory.jsonBuilder()
            .startObject()
            .field(FIELD_NAME, Floats.asList(testData.indexData.vectors[0]).toArray())
            .field("description", "Test document")
            .endObject()
            .toString();
        addKnnDoc(indexName, docId, docWithBoth);

        // Update doc to remove vector field, keeping only text field
        addNonKNNDoc(indexName, docId, "description", "Updated test document");

        flush(indexName, true);

        // Verify index has one doc with no vector field and search returns no results
        assertEquals(1, getDocCount(indexName));
        List<KNNResult> results = runKnnQuery(indexName, FIELD_NAME, testData.queries[0], 5);
        assertEquals(0, results.size());
    }

    private float getRecall(final Set<String> truth, final Set<String> result) {
        // Count the number of relevant documents retrieved
        result.retainAll(truth);
        int relevantRetrieved = result.size();

        // Total number of relevant documents
        int totalRelevant = truth.size();

        // Calculate recall
        return (float) relevantRetrieved / totalRelevant;
    }

    private List<KNNResult> runKnnQuery(final String indexName, final String fieldName, final float[] queryVector, final int k)
        throws Exception {
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(ArrayUtils.toObject(queryVector))
            .k(k)
            .build()
            .getQueryString();
        Response response = searchKNNIndex(indexName, query, k);
        return parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
    }

    private void ingestTestData(final String indexName, final String fieldName) throws Exception {
        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                fieldName,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Assert we have the right number of documents in the index
        refreshAllIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));
    }

    private void createKnnHnswIndex(final KNNEngine knnEngine, final String indexName, final String fieldName, final int dimension)
        throws IOException {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.L2.getValue())
            .engine(knnEngine.getName())
            .build();

        String knnIndexMapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(fieldName)
            .dimension(dimension)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        createKnnIndex(indexName, knnIndexMapping);
    }
}
