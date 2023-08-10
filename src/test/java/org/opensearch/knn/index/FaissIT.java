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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class FaissIT extends KNNRestTestCase {
    private static final String DOC_ID_1 = "doc1";
    private static final String DOC_ID_2 = "doc2";
    private static final String DOC_ID_3 = "doc3";
    private static final String COLOR_FIELD_NAME = "color";
    private static final String TASTE_FIELD_NAME = "taste";

    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        URL testIndexVectors = FaissIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = FaissIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    public void testEndToEnd_fromHNSWMethod() throws Exception {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, hnswMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

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
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, testData.queries[i], k), k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = Floats.toArray(Arrays.stream(knnResults.get(j).getVector()).collect(Collectors.toList()));
                assertEquals(
                    KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
        }

        // Delete index
        deleteKNNIndex(indexName);

        // Search every 5 seconds 14 times to confirm graph gets evicted
        int intervals = 14;
        for (int i = 0; i < intervals; i++) {
            if (getTotalGraphsInCache() == 0) {
                return;
            }

            Thread.sleep(5 * 1000);
        }

        fail("Graphs are not getting evicted");
    }

    public void testEndToEnd_fromNSGMethod() throws Exception {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";

        KNNMethod nsgMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_NSG);
        SpaceType spaceType = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, nsgMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_R, 64)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, mapping);

        // Index the test data
        bulkAddKnnDocs(indexName, fieldName, testData.indexData.vectors, testData.indexData.docs.length);

        // Assert we have the right number of documents in the index
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, testData.queries[i], k), k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = Floats.toArray(Arrays.stream(knnResults.get(j).getVector()).collect(Collectors.toList()));
                assertEquals(
                    KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
        }

        // Delete index
        deleteKNNIndex(indexName);

        // Search every 5 seconds 14 times to confirm graph gets evicted
        int intervals = 14;
        for (int i = 0; i < intervals; i++) {
            if (getTotalGraphsInCache() == 0) {
                return;
            }

            Thread.sleep(5 * 1000);
        }

        fail("Graphs are not getting evicted");
    }

    public void testEndToEnd_fromNSGMethodIndexOne() throws Exception {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";

        KNNMethod nsgMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_NSG);
        SpaceType spaceType = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, nsgMethod.getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_R, 64)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, mapping);
        Map<String, Object> maping = getIndexMappingAsMap(indexName);
        // assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(maping));

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
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        int k = 10;
        for (int i = 0; i < testData.queries.length; i++) {
            Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, testData.queries[i], k), k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = Floats.toArray(Arrays.stream(knnResults.get(j).getVector()).collect(Collectors.toList()));
                assertEquals(
                    KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
        }

        // Delete index
        deleteKNNIndex(indexName);

        // Search every 5 seconds 14 times to confirm graph gets evicted
        int intervals = 14;
        for (int i = 0; i < intervals; i++) {
            if (getTotalGraphsInCache() == 0) {
                return;
            }

            Thread.sleep(5 * 1000);
        }

        fail("Graphs are not getting evicted");
    }

    public void testDocUpdate() throws IOException {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";
        Integer dimension = 2;

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L2;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
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
        createKnnIndex(indexName, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // update
        Float[] updatedVector = { 8.0f, 8.0f };
        updateKnnDoc(INDEX_NAME, "1", FIELD_NAME, updatedVector);

    }

    public void testDocDeletion() throws IOException {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";
        Integer dimension = 2;

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.L2;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
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
        createKnnIndex(indexName, mapping);

        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // delete knn doc
        deleteKnnDoc(INDEX_NAME, "1");
    }

    @SneakyThrows
    public void testKNNQuery_withModelDifferentCombination_thenSuccess() {

        String modelId = "test-model";
        int dimension = 128;

        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";

        // Add training data
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        int trainingDataCount = 200;
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        // Call train API - IVF with nlists = 1 is brute force, but will require training
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, "ivf")
            .field(KNN_ENGINE, "faiss")
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, "faiss test description");

        // Make sure training succeeds after 30 seconds
        assertTrainingSucceeds(modelId, 30, 1000);

        // Create knn index from model
        String fieldName = "test-field-name";
        String indexName = "test-index-name";
        String indexMapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field(MODEL_ID, modelId)
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(indexName, getKNNDefaultIndexSettings(), indexMapping);

        // Index some documents
        int numDocs = 100;
        for (int i = 0; i < numDocs; i++) {
            float[] indexVector = new float[dimension];
            Arrays.fill(indexVector, (float) i);
            addKnnDocWithAttributes(indexName, Integer.toString(i), fieldName, indexVector, ImmutableMap.of("rating", String.valueOf(i)));
        }

        // Run search and ensure that the values returned are expected
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);
        int k = 10;

        Response searchResponse = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, queryVector, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);

        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
        }

        // doing exact search with filters
        Response exactSearchFilteredResponse = searchKNNIndex(
            indexName,
            new KNNQueryBuilder(fieldName, queryVector, k, QueryBuilders.rangeQuery("rating").gte("90").lte("99")),
            k
        );
        List<KNNResult> exactSearchFilteredResults = parseSearchResponse(
            EntityUtils.toString(exactSearchFilteredResponse.getEntity()),
            fieldName
        );
        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(exactSearchFilteredResults.get(i).getDocId()));
        }

        // doing exact search with filters
        Response aNNSearchFilteredResponse = searchKNNIndex(
            indexName,
            new KNNQueryBuilder(fieldName, queryVector, k, QueryBuilders.rangeQuery("rating").gte("80").lte("99")),
            k
        );
        List<KNNResult> aNNSearchFilteredResults = parseSearchResponse(
            EntityUtils.toString(aNNSearchFilteredResponse.getEntity()),
            fieldName
        );
        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(aNNSearchFilteredResults.get(i).getDocId()));
        }
    }

    @SneakyThrows
    public void testQueryWithFilter_withDifferentCombination_thenSuccess() {
        setupKNNIndexForFilterQuery();
        final float[] searchVector = { 6.0f, 6.0f, 4.1f };
        // K > filteredResults
        int kGreaterThanFilterResult = 5;
        List<String> expectedDocIds = Arrays.asList(DOC_ID_1, DOC_ID_3);
        final Response response = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, kGreaterThanFilterResult, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
            kGreaterThanFilterResult
        );
        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(expectedDocIds.size(), knnResults.size());
        assertTrue(knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toList()).containsAll(expectedDocIds));

        // K Limits Filter results
        int kLimitsFilterResult = 1;
        List<String> expectedDocIdsKLimitsFilterResult = List.of(DOC_ID_1);
        final Response responseKLimitsFilterResult = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, kLimitsFilterResult, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
            kLimitsFilterResult
        );
        final String responseBodyKLimitsFilterResult = EntityUtils.toString(responseKLimitsFilterResult.getEntity());
        final List<KNNResult> knnResultsKLimitsFilterResult = parseSearchResponse(responseBodyKLimitsFilterResult, FIELD_NAME);

        assertEquals(expectedDocIdsKLimitsFilterResult.size(), knnResultsKLimitsFilterResult.size());
        assertTrue(
            knnResultsKLimitsFilterResult.stream()
                .map(KNNResult::getDocId)
                .collect(Collectors.toList())
                .containsAll(expectedDocIdsKLimitsFilterResult)
        );

        // Empty filter docIds
        int k = 10;
        final Response emptyFilterResponse = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(
                FIELD_NAME,
                searchVector,
                kLimitsFilterResult,
                QueryBuilders.termQuery(COLOR_FIELD_NAME, "color_not_present")
            ),
            k
        );
        final String responseBodyForEmptyDocIds = EntityUtils.toString(emptyFilterResponse.getEntity());
        final List<KNNResult> emptyKNNFilteredResultsFromResponse = parseSearchResponse(responseBodyForEmptyDocIds, FIELD_NAME);

        assertEquals(0, emptyKNNFilteredResultsFromResponse.size());
    }

    protected void setupKNNIndexForFilterQuery() throws Exception {
        // Create Mappings
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 3)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW).getMethodComponent().getName())
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        final String mapping = builder.toString();

        createKnnIndex(INDEX_NAME, mapping);

        addKnnDocWithAttributes(
            DOC_ID_1,
            new float[] { 6.0f, 7.9f, 3.1f },
            ImmutableMap.of(COLOR_FIELD_NAME, "red", TASTE_FIELD_NAME, "sweet")
        );
        addKnnDocWithAttributes(DOC_ID_2, new float[] { 3.2f, 2.1f, 4.8f }, ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes(DOC_ID_3, new float[] { 4.1f, 5.0f, 7.1f }, ImmutableMap.of(COLOR_FIELD_NAME, "red"));

        refreshIndex(INDEX_NAME);
    }
}
