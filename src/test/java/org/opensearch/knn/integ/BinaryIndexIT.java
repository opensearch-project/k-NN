/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.collect.ImmutableList;
import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.ArrayUtils;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * This class contains integration tests for binary index with HNSW in Faiss
 */
@Log4j2
public class BinaryIndexIT extends KNNRestTestCase {
    private static TestUtils.TestData testData;
    private static final int NEVER_BUILD_GRAPH = -1;
    private static final int ALWAYS_BUILD_GRAPH = 0;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (BinaryIndexIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of BinaryIndexIT Class is null");
        }
        URL testIndexVectors = BinaryIndexIT.class.getClassLoader().getResource("data/test_vectors_binary_1000x128.json");
        URL testQueries = BinaryIndexIT.class.getClassLoader().getResource("data/test_queries_binary_100x128.csv");
        URL groundTruthValues = BinaryIndexIT.class.getClassLoader().getResource("data/test_ground_truth_binary_100.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        assert groundTruthValues != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath(), groundTruthValues.getPath());
    }

    @After
    public void cleanUp() {
        try {
            deleteKNNIndex(INDEX_NAME);
        } catch (Exception e) {
            log.error(e);
        }
    }

    @SneakyThrows
    public void testFaissHnswBinary_whenSmallDataSet_thenCreateIngestQueryWorks() {
        // Create Index
        createKnnHnswBinaryIndex(KNNEngine.FAISS, INDEX_NAME, FIELD_NAME, 16);

        // Ingest
        Byte[] vector1 = { 0b00000001, 0b00000001 };
        Byte[] vector2 = { 0b00000011, 0b00000001 };
        Byte[] vector3 = { 0b00000111, 0b00000001 };
        Byte[] vector4 = { 0b00001111, 0b00000001 };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector1);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, vector2);
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, vector3);
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, vector4);

        // Query
        float[] queryVector = { (byte) 0b10001111, (byte) 0b10000000 };
        int k = 4;
        List<KNNResult> results = runKnnQuery(INDEX_NAME, FIELD_NAME, queryVector, k);

        // Validate
        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(k - i, Integer.parseInt(results.get(i).getDocId()));
        }
    }

    @SneakyThrows
    public void testFaissHnswBinary_when1000Data_thenRecallIsAboveNinePointZero() {
        // Create Index
        createKnnHnswBinaryIndex(KNNEngine.FAISS, INDEX_NAME, FIELD_NAME, 128);
        ingestTestData(INDEX_NAME, FIELD_NAME);

        int k = 100;
        for (int i = 0; i < testData.queries.length; i++) {
            List<KNNResult> knnResults = runKnnQuery(INDEX_NAME, FIELD_NAME, testData.queries[i], k);
            float recall = getRecall(
                Set.of(Arrays.copyOf(testData.groundTruthValues[i], k)),
                knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet())
            );
            assertTrue("Recall: " + recall, recall > 0.1);
        }
    }

    @SneakyThrows
    public void testFaissHnswBinary_whenBuildVectorGraphThresholdIsNegativeEndToEnd_thenBuildGraphBasedOnSetting() {
        // Create Index
        createKnnHnswBinaryIndex(KNNEngine.FAISS, INDEX_NAME, FIELD_NAME, 128, NEVER_BUILD_GRAPH);
        ingestTestData(INDEX_NAME, FIELD_NAME);

        assertEquals(1, runKnnQuery(INDEX_NAME, FIELD_NAME, testData.queries[0], 1).size());

        // update build vector data structure setting
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD, 0));
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 100;
        for (int i = 0; i < testData.queries.length; i++) {
            List<KNNResult> knnResults = runKnnQuery(INDEX_NAME, FIELD_NAME, testData.queries[i], k);
            float recall = getRecall(
                Set.of(Arrays.copyOf(testData.groundTruthValues[i], k)),
                knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet())
            );
            assertTrue("Recall: " + recall, recall > 0.1);
        }
    }

    @SneakyThrows
    public void testFaissHnswBinary_whenBuildVectorGraphThresholdIsProvidedEndToEnd_thenBuildGraphBasedOnSetting() throws Exception {
        // Create Index
        createKnnHnswBinaryIndex(KNNEngine.FAISS, INDEX_NAME, FIELD_NAME, 128, testData.indexData.docs.length);
        ingestTestData(INDEX_NAME, FIELD_NAME, false);

        assertEquals(1, runKnnQuery(INDEX_NAME, FIELD_NAME, testData.queries[0], 1).size());

        // update build vector data structure setting
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD, 0));
        forceMergeKnnIndex(INDEX_NAME, 1);

        int k = 100;
        for (int i = 0; i < testData.queries.length; i++) {
            List<KNNResult> knnResults = runKnnQuery(INDEX_NAME, FIELD_NAME, testData.queries[i], k);
            float recall = getRecall(
                Set.of(Arrays.copyOf(testData.groundTruthValues[i], k)),
                knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet())
            );
            assertTrue("Recall: " + recall, recall > 0.1);
        }
    }

    @SneakyThrows
    public void testFaissHnswBinary_whenRadialSearch_thenThrowException() {
        // Create Index
        createKnnHnswBinaryIndex(KNNEngine.FAISS, INDEX_NAME, FIELD_NAME, 16);

        // Query
        float[] queryVector = { (byte) 0b10001111, (byte) 0b10000000 };
        Exception e = expectThrows(Exception.class, () -> runRnnQuery(INDEX_NAME, FIELD_NAME, queryVector, 1, 4));
        assertTrue(e.getMessage(), e.getMessage().contains("Binary data type does not support radial search"));
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

    private List<KNNResult> runRnnQuery(
        final String indexName,
        final String fieldName,
        final float[] queryVector,
        final float minScore,
        final int size
    ) throws Exception {
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(ArrayUtils.toObject(queryVector))
            .minScore(minScore)
            .build()
            .getQueryString();
        Response response = searchKNNIndex(indexName, query, size);
        return parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
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
        ingestTestData(indexName, fieldName, true);
    }

    private void ingestTestData(final String indexName, final String fieldName, boolean refresh) throws Exception {
        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                ImmutableList.of(fieldName),
                ImmutableList.of(Floats.asList(testData.indexData.vectors[i]).toArray()),
                refresh
            );
        }

        // Assert we have the right number of documents in the index
        refreshAllIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));
    }

    private void createKnnHnswBinaryIndex(
        final KNNEngine knnEngine,
        final String indexName,
        final String fieldName,
        final int dimension,
        final int threshold
    ) throws IOException {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .engine(knnEngine.getName())
            .build();

        String knnIndexMapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(fieldName)
            .dimension(dimension)
            .vectorDataType(VectorDataType.BINARY.getValue())
            .method(method)
            .build()
            .getIndexMapping();
        createKnnIndex(indexName, buildKNNIndexSettings(threshold), knnIndexMapping);
    }

    private void createKnnHnswBinaryIndex(final KNNEngine knnEngine, final String indexName, final String fieldName, final int dimension)
        throws IOException {
        createKnnHnswBinaryIndex(knnEngine, indexName, fieldName, dimension, ALWAYS_BUILD_GRAPH);
    }
}
