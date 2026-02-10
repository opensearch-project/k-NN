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

import com.carrotsearch.randomizedtesting.annotations.TimeoutSuite;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.lucene.tests.util.TimeUnits;
import org.apache.lucene.util.VectorUtil;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.client.ResponseException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.MatchNoneQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.RangeQueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.FP16_MAX_VALUE;
import static org.opensearch.knn.common.KNNConstants.FP16_MIN_VALUE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

@TimeoutSuite(millis = 40 * TimeUnits.MINUTE)
public class FaissIT extends KNNRestTestCase {
    private static final String DOC_ID_1 = "doc1";
    private static final String DOC_ID_2 = "doc2";
    private static final String DOC_ID_3 = "doc3";
    private static final String COLOR_FIELD_NAME = "color";
    private static final String TASTE_FIELD_NAME = "taste";

    private static final String DIMENSION_FIELD_NAME = "dimension";
    private static final int VECTOR_DIMENSION = 3;
    private static final String KNN_VECTOR_TYPE = "knn_vector";
    private static final String PROPERTIES_FIELD_NAME = "properties";
    private static final String TYPE_FIELD_NAME = "type";
    private static final String INTEGER_FIELD_NAME = "int_field";
    private static final String FILED_TYPE_INTEGER = "integer";
    private static final String NON_EXISTENT_INTEGER_FIELD_NAME = "nonexistent_int_field";
    public static final int NEVER_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD = -1;
    public static final int ALWAYS_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD = 0;

    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (FaissIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of FaissIT Class is null");
        }
        URL testIndexVectors = FaissIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = FaissIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testEndToEnd_whenDoRadiusSearch_whenDistanceThreshold_whenMethodIsHNSWFlat_thenSucceed() {
        SpaceType spaceType = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(INDEX_NAME, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(INDEX_NAME)));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(testData.indexData.docs[i]),
                FIELD_NAME,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Assert we have the right number of documents
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(INDEX_NAME));

        float distance = 300000000000f;
        validateRadiusSearchResults(INDEX_NAME, FIELD_NAME, testData.queries, distance, null, spaceType, null, null);

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testEndToEnd_whenDoRadiusSearch_whenScoreThreshold_whenMethodIsHNSWFlat_thenSucceed() {
        SpaceType spaceType = SpaceType.L2;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(INDEX_NAME, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(INDEX_NAME)));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(testData.indexData.docs[i]),
                FIELD_NAME,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Assert we have the right number of documents
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(INDEX_NAME));

        float score = 0.00001f;

        validateRadiusSearchResults(INDEX_NAME, FIELD_NAME, testData.queries, null, score, spaceType, null, null);

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testRadialSearchWithCosineAndFilter_ensureThresholdEnforced_thenSucceed() {
        String indexName = "test-index-cosine-radial";
        String filterFieldName = "category";
        int dimension = 128;
        SpaceType spaceType = SpaceType.COSINESIMIL;

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, 16)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 128)
            .endObject()
            .endObject()
            .endObject()
            .startObject(filterFieldName)
            .field(TYPE_FIELD_NAME, "keyword")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, builder.toString());

        Random random = new Random();
        float[] baseVector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            baseVector[i] = random.nextFloat() - 0.5f;
        }
        VectorUtil.l2normalize(baseVector);

        for (int i = 0; i < 30; i++) {
            float[] vector = new float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = baseVector[j] + (random.nextFloat() - 0.5f) * 0.3f;
            }
            VectorUtil.l2normalize(vector);
            addKnnDocWithAttributes(indexName, "doc_" + i, FIELD_NAME, vector, ImmutableMap.of(filterFieldName, "A"));
        }

        refreshIndex(indexName);
        forceMergeKnnIndex(indexName);

        float[] queryVector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            queryVector[i] = baseVector[i] + (random.nextFloat() - 0.5f) * 0.1f;
        }
        VectorUtil.l2normalize(queryVector);

        float minScore = 0.935f;
        TermQueryBuilder filterQuery = QueryBuilders.termQuery(filterFieldName, "A");

        List<List<KNNResult>> results = validateRadiusSearchResults(
            indexName,
            FIELD_NAME,
            new float[][] { queryVector },
            null,
            minScore,
            spaceType,
            filterQuery,
            null
        );

        for (KNNResult result : results.get(0)) {
            assertTrue(String.format("Score %.4f below threshold %.3f", result.getScore(), minScore), result.getScore() >= minScore);
        }

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testEndToEnd_whenDoRadiusSearch_whenMoreThanOneScoreThreshold_whenMethodIsHNSWFlat_thenSucceed() {
        SpaceType spaceType = SpaceType.INNER_PRODUCT;

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(INDEX_NAME, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(INDEX_NAME)));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(testData.indexData.docs[i]),
                FIELD_NAME,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Assert we have the right number of documents
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(INDEX_NAME));

        float score = 5f;

        validateRadiusSearchResults(INDEX_NAME, FIELD_NAME, testData.queries, null, score, spaceType, null, null);

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testEndToEnd_whenDoRadiusSearch_whenDistanceThreshold_whenMethodIsHNSWPQ_thenSucceed() {
        String indexName = "test-index";
        String fieldName = "test-field";
        String trainingIndexName = "training-index";
        String trainingFieldName = "training-field";

        String modelId = "test-model";
        String modelDescription = "test model";

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> pqMValues = ImmutableList.of(2, 4, 8);

        // training data needs to be at least equal to the number of centroids for PQ
        // which is 2^8 = 256. 8 because that's the only valid code_size for HNSWPQ
        int trainingDataCount = 1100;

        SpaceType spaceType = SpaceType.L2;

        int dimension = testData.indexData.vectors[0].length;

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_PQ)
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_M, pqMValues.get(random().nextInt(pqMValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);

        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        ingestDataAndTrainModel(modelId, trainingIndexName, trainingFieldName, dimension, modelDescription, in, trainingDataCount);
        assertTrainingSucceeds(modelId, 360, 1000);

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("model_id", modelId)
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

        float distance = 300000000000f;
        // create method parameter wih ef_search
        Map<String, Object> methodParameters = new ImmutableMap.Builder<String, Object>().put(KNNConstants.METHOD_PARAMETER_EF_SEARCH, 150)
            .build();

        validateRadiusSearchResults(indexName, fieldName, testData.queries, distance, null, spaceType, null, methodParameters);

        // Delete index
        deleteKNNIndex(indexName);
        deleteModel(modelId);

        // Search every 5 seconds 14 times to confirm graph gets evicted
        validateGraphEviction();
    }

    @SneakyThrows
    public void testRadialQuery_withFilter_thenSuccess() {
        setupKNNIndexForFilterQuery();

        final float[][] searchVector = new float[][] { { 3.3f, 3.0f, 5.0f } };
        TermQueryBuilder termQueryBuilder = QueryBuilders.termQuery("color", "red");
        List<String> expectedDocIds = Arrays.asList(DOC_ID_3);

        float distance = 15f;
        List<List<KNNResult>> queryResult = validateRadiusSearchResults(
            INDEX_NAME,
            FIELD_NAME,
            searchVector,
            distance,
            null,
            SpaceType.L2,
            termQueryBuilder,
            null
        );

        assertEquals(1, queryResult.get(0).size());
        assertEquals(expectedDocIds.get(0), queryResult.get(0).get(0).getDocId());

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testQueryWithFilterMultipleShards() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, "3")
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject(INTEGER_FIELD_NAME)
            .field(TYPE_FIELD_NAME, FILED_TYPE_INTEGER)
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();

        createIndex(INDEX_NAME, Settings.builder().put("number_of_shards", 10).put("number_of_replicas", 0).put("index.knn", true).build());
        putMappingRequest(INDEX_NAME, mapping);

        addKnnDocWithAttributes("doc1", new float[] { 7.0f, 7.0f, 3.0f }, ImmutableMap.of("dateReceived", "2024-10-01"));

        refreshIndex(INDEX_NAME);

        final float[] searchVector = { 6.0f, 7.0f, 3.0f };
        final Response response = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(
                FIELD_NAME,
                searchVector,
                1,
                QueryBuilders.boolQuery().must(QueryBuilders.rangeQuery("dateReceived").gte("2023-11-01"))
            ),
            10
        );
        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(1, knnResults.size());
    }

    @SneakyThrows
    public void testQueryWithFilterFunctionAppliedMultipleShards() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, "3")
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject(INTEGER_FIELD_NAME)
            .field(TYPE_FIELD_NAME, FILED_TYPE_INTEGER)
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        createIndex(INDEX_NAME, Settings.builder().put("number_of_shards", 10).put("number_of_replicas", 0).put("index.knn", true).build());
        putMappingRequest(INDEX_NAME, mapping);

        addKnnDocWithAttributes("doc1", new float[] { 7.0f, 7.0f, 3.0f }, ImmutableMap.of("dateReceived", "2024-10-01"));

        refreshIndex(INDEX_NAME);

        final float[] searchVector = { 6.0f, 7.0f, 3.0f };

        // Add initial RangeQuery to a new KNNQueryBuilder
        RangeQueryBuilder rangeQueryBuilder = QueryBuilders.rangeQuery("dateReceived").gte("2023-11-01");
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, searchVector, 1, null);
        KNNQueryBuilder updatedKnnQueryBuilder = (KNNQueryBuilder) knnQueryBuilder.filter(rangeQueryBuilder);
        Response response = searchKNNIndex(INDEX_NAME, updatedKnnQueryBuilder, 10);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(1, knnResults.size());

        // Apply another MatchNoneQueryBuilder to new KNNQueryBuilder
        updatedKnnQueryBuilder = (KNNQueryBuilder) knnQueryBuilder.filter(new MatchNoneQueryBuilder());
        response = searchKNNIndex(INDEX_NAME, updatedKnnQueryBuilder, 10);
        responseBody = EntityUtils.toString(response.getEntity());
        knnResults = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(0, knnResults.size());
    }

    @SneakyThrows
    public void testEndToEnd_whenMethodIsHNSWPQ_thenSucceed() {
        String indexName = "test-index";
        String fieldName = "test-field";
        String trainingIndexName = "training-index";
        String trainingFieldName = "training-field";

        String modelId = "test-model";
        String modelDescription = "test model";

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> pqMValues = ImmutableList.of(2, 4, 8);

        // training data needs to be at least equal to the number of centroids for PQ
        // which is 2^8 = 256. 8 because thats the only valid code_size for HNSWPQ
        int trainingDataCount = 1100;

        SpaceType spaceType = SpaceType.L2;

        Integer dimension = testData.indexData.vectors[0].length;

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_PQ)
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_M, pqMValues.get(random().nextInt(pqMValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);

        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        ingestDataAndTrainModel(modelId, trainingIndexName, trainingFieldName, dimension, modelDescription, in, trainingDataCount);
        assertTrainingSucceeds(modelId, 360, 1000);

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("model_id", modelId)
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
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
        }

        // Delete index
        deleteKNNIndex(indexName);
        deleteModel(modelId);

        validateGraphEviction();
    }

    @SneakyThrows
    public void testHNSWSQFP16_whenIndexedAndQueried_thenSucceed() {
        String indexName = "test-index-hnsw-sqfp16";
        String fieldName = "test-field-hnsw-sqfp16";
        SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.INNER_PRODUCT };
        Random random = new Random();
        SpaceType spaceType = spaceTypes[random.nextInt(spaceTypes.length)];

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        int dimension = 128;
        int numDocs = 100;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));
        indexTestData(indexName, fieldName, dimension, numDocs);
        queryTestData(indexName, fieldName, dimension, numDocs);
        deleteKNNIndex(indexName);
        validateGraphEviction();
    }

    @SneakyThrows
    public void testHNSWSQFP16_whenGraphThresholdIsNegative_whenIndexed_thenSkipCreatingGraph() {
        final String indexName = "test-index-hnsw-sqfp16";
        final String fieldName = "test-field-hnsw-sqfp16";
        final SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.INNER_PRODUCT };
        final Random random = new Random();
        final SpaceType spaceType = spaceTypes[random.nextInt(spaceTypes.length)];

        final int dimension = 128;
        final int numDocs = 100;

        // Create an index
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        final Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        final String mapping = builder.toString();
        final Settings knnIndexSettings = buildKNNIndexSettings(-1);
        createKnnIndex(indexName, knnIndexSettings, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));
        indexTestData(indexName, fieldName, dimension, numDocs);

        final float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);

        // Assert we have the right number of documents in the index
        assertEquals(numDocs, getDocCount(indexName));

        final Response searchResponse = searchKNNIndex(indexName, buildSearchQuery(fieldName, 1, queryVector, null), 1);
        final List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        // expect result due to exact search
        assertEquals(1, results.size());

        deleteKNNIndex(indexName);
        validateGraphEviction();
    }

    @SneakyThrows
    public void testHNSWSQFP16_whenGraphThresholdIsMetDuringMerge_thenCreateGraph() {
        final String indexName = "test-index-hnsw-sqfp16";
        final String fieldName = "test-field-hnsw-sqfp16";
        final SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.INNER_PRODUCT };
        final Random random = new Random();
        final SpaceType spaceType = spaceTypes[random.nextInt(spaceTypes.length)];
        final int dimension = 128;
        final int numDocs = 100;

        // Create an index
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        final Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        final String mapping = builder.toString();
        final Settings knnIndexSettings = buildKNNIndexSettings(numDocs);
        createKnnIndex(indexName, knnIndexSettings, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));
        indexTestData(indexName, fieldName, dimension, numDocs);

        final float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);

        // Assert we have the right number of documents in the index
        assertEquals(numDocs, getDocCount(indexName));

        // KNN Query should return empty result
        final Response searchResponse = searchKNNIndex(indexName, buildSearchQuery(fieldName, 1, queryVector, null), 1);
        final List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(1, results.size());

        // update index setting to build graph and do force merge
        // update build vector data structure setting
        forceMergeKnnIndex(indexName, 1);

        queryTestData(indexName, fieldName, dimension, numDocs);

        deleteKNNIndex(indexName);
        validateGraphEviction();
    }

    @SneakyThrows
    public void testIVFSQFP16_whenIndexedAndQueried_thenSucceed() {

        String modelId = "test-model-ivf-sqfp16";
        int dimension = 128;
        int numDocs = 100;

        String trainingIndexName = "train-index-ivf-sqfp16";
        String trainingFieldName = "train-field-ivf-sqfp16";

        // Add training data
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        int trainingDataCount = 1100;
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NPROBES, 4)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, "faiss ivf sqfp16 test description");

        // Make sure training succeeds after 30 seconds
        assertTrainingSucceeds(modelId, 30, 1000);

        // Create knn index from model
        String fieldName = "test-field-name-ivf-sqfp16";
        String indexName = "test-index-name-ivf-sqfp16";
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

        indexTestData(indexName, fieldName, dimension, numDocs);
        queryTestData(indexName, fieldName, dimension, numDocs);
        queryTestData(indexName, fieldName, dimension, numDocs, Map.of("nprobes", 100));
        deleteKNNIndex(indexName);
        validateGraphEviction();
    }

    @SneakyThrows
    public void testHNSWSQFP16_whenIndexedWithOutOfFP16Range_thenThrowException() {
        String indexName = "test-index-sqfp16";
        String fieldName = "test-field-sqfp16";
        SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.INNER_PRODUCT };
        Random random = new Random();
        SpaceType spaceType = spaceTypes[random.nextInt(spaceTypes.length)];

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        int dimension = 128;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

        Float[] vector = new Float[dimension];
        Float[] vector1 = new Float[dimension];
        Float[] vector2 = new Float[dimension];
        int halfDimension = dimension / 2;

        for (int i = 0; i < dimension; i++) {
            if (i < halfDimension) {
                vector[i] = -10.76f;
                vector1[i] = -65506.84f;
                vector2[i] = -65526.4567f;
            } else {
                vector[i] = 65504.2f;
                vector1[i] = 12.56f;
                vector2[i] = 65526.4567f;
            }
        }

        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(indexName, "1", fieldName, vector));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );

        ResponseException ex1 = expectThrows(ResponseException.class, () -> addKnnDoc(indexName, "2", fieldName, vector1));
        assertTrue(
            ex1.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );

        ResponseException ex2 = expectThrows(ResponseException.class, () -> addKnnDoc(indexName, "3", fieldName, vector2));
        assertTrue(
            ex2.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );
        deleteKNNIndex(indexName);
        validateGraphEviction();
    }

    @SneakyThrows
    public void testHNSWSQFP16_whenClipToFp16isTrueAndIndexedWithOutOfFP16Range_thenSucceed() {
        String indexName = "test-index-sqfp16-clip-fp16";
        String fieldName = "test-field-sqfp16";

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        int dimension = 128;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .field(FAISS_SQ_CLIP, true)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

        Float[] vector1 = new Float[dimension];
        Float[] vector2 = new Float[dimension];
        Float[] vector3 = new Float[dimension];
        Float[] vector4 = new Float[dimension];
        float[] queryVector = new float[dimension];
        int halfDimension = dimension / 2;

        for (int i = 0; i < dimension; i++) {
            if (i < halfDimension) {
                vector1[i] = -65523.76f;
                vector2[i] = -270.85f;
                vector3[i] = -150.9f;
                vector4[i] = -20.89f;
                queryVector[i] = -10.5f;
            } else {
                vector1[i] = 65504.2f;
                vector2[i] = 65514.2f;
                vector3[i] = 65504.0f;
                vector4[i] = 100000000.0f;
                queryVector[i] = 25.48f;
            }
        }

        addKnnDoc(indexName, "1", fieldName, vector1);
        addKnnDoc(indexName, "2", fieldName, vector2);
        addKnnDoc(indexName, "3", fieldName, vector3);
        addKnnDoc(indexName, "4", fieldName, vector4);

        int k = 4;
        Response searchResponse = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, queryVector, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(k - i, Integer.parseInt(results.get(i).getDocId()));
        }

        deleteKNNIndex(indexName);
        validateGraphEviction();
    }

    @SneakyThrows
    public void testIVFSQFP16_whenIndexedWithOutOfFP16Range_thenThrowException() {
        String modelId = "test-model-ivf-sqfp16";
        int dimension = 128;

        String trainingIndexName = "train-index-ivf-sqfp16";
        String trainingFieldName = "train-field-ivf-sqfp16";

        // Add training data
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        int trainingDataCount = 1100;
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, "faiss ivf sqfp16 test description");

        // Make sure training succeeds after 30 seconds
        assertTrainingSucceeds(modelId, 30, 1000);

        // Create knn index from model
        String fieldName = "test-field-name-ivf-sqfp16";
        String indexName = "test-index-name-ivf-sqfp16";
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
        Float[] vector = { -10.76f, 65504.2f };

        ResponseException ex = expectThrows(ResponseException.class, () -> addKnnDoc(indexName, "1", fieldName, vector));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );

        Float[] vector1 = { -65506.84f, 12.56f };

        ResponseException ex1 = expectThrows(ResponseException.class, () -> addKnnDoc(indexName, "2", fieldName, vector1));
        assertTrue(
            ex1.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );

        Float[] vector2 = { -65526.4567f, 65526.4567f };

        ResponseException ex2 = expectThrows(ResponseException.class, () -> addKnnDoc(indexName, "3", fieldName, vector2));
        assertTrue(
            ex2.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );
        deleteKNNIndex(indexName);
        deleteKNNIndex(trainingIndexName);
        deleteModel(modelId);
    }

    @SneakyThrows
    public void testIVFSQFP16_whenClipToFp16isTrueAndIndexedWithOutOfFP16Range_thenSucceed() {
        String modelId = "test-model-ivf-sqfp16";
        int dimension = 2;

        String trainingIndexName = "train-index-ivf-sqfp16";
        String trainingFieldName = "train-field-ivf-sqfp16";

        // Add training data
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        int trainingDataCount = 1100;
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .field(FAISS_SQ_CLIP, true)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, "faiss ivf sqfp16 test description");

        // Make sure training succeeds after 30 seconds
        assertTrainingSucceeds(modelId, 30, 1000);

        // Create knn index from model
        String fieldName = "test-field-name-ivf-sqfp16";
        String indexName = "test-index-name-ivf-sqfp16";
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
        Float[] vector1 = { -65523.76f, 65504.2f };
        Float[] vector2 = { -270.85f, 65514.2f };
        Float[] vector3 = { -150.9f, 65504.0f };
        Float[] vector4 = { -20.89f, 100000000.0f };
        addKnnDoc(indexName, "1", fieldName, vector1);
        addKnnDoc(indexName, "2", fieldName, vector2);
        addKnnDoc(indexName, "3", fieldName, vector3);
        addKnnDoc(indexName, "4", fieldName, vector4);

        float[] queryVector = { -10.5f, 25.48f };
        int k = 4;
        Response searchResponse = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, queryVector, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(k - i, Integer.parseInt(results.get(i).getDocId()));
        }

        deleteKNNIndex(indexName);
        deleteKNNIndex(trainingIndexName);
        deleteModel(modelId);
        validateGraphEviction();
    }

    @SneakyThrows
    public void testEndToEnd_whenMethodIsHNSWPQAndHyperParametersNotSet_thenSucceed() {
        String indexName = "test-index";
        String fieldName = "test-field";
        String trainingIndexName = "training-index";
        String trainingFieldName = "training-field";

        String modelId = "test-model";
        String modelDescription = "test model";

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> pqMValues = ImmutableList.of(2, 4, 8);

        // training data needs to be at least equal to the number of centroids for PQ
        // which is 2^8 = 256. 8 because thats the only valid code_size for HNSWPQ
        int trainingDataCount = 1100;

        SpaceType spaceType = SpaceType.L2;

        Integer dimension = testData.indexData.vectors[0].length;

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_PQ)
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_M, pqMValues.get(random().nextInt(pqMValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);

        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        ingestDataAndTrainModel(modelId, trainingIndexName, trainingFieldName, dimension, modelDescription, in, trainingDataCount);
        assertTrainingSucceeds(modelId, 360, 1000);

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("model_id", modelId)
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
                float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    KNNEngine.FAISS.score(KNNScoringUtil.l2Squared(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
        }

        // Delete index
        deleteKNNIndex(indexName);
        deleteModel(modelId);

        validateGraphEviction();
    }

    /**
     * This test confirms that sharing index state for IVFPQ-l2 indices functions properly. The main functionality that
     * needs to be confirmed is that once an index gets deleted, it will not cause a failure for the non-deleted index.
     *
     * The workflow will be:
     * 1. Create a model
     * 2. Create two indices index from the model
     * 3. Load the native index files from the first index
     * 4. Assert search works
     * 5. Load the native index files (which will reuse the shared state from the initial index)
     * 6. Assert search works on the second index
     * 7. Delete the first index and wait
     * 8. Assert search works on the second index
     */
    @SneakyThrows
    public void testSharedIndexState_whenOneIndexDeleted_thenSecondIndexIsStillSearchable() {
        String firstIndexName = "test-index-1";
        String secondIndexName = "test-index-2";
        String trainingIndexName = "training-index";

        String modelId = "test-model";
        String modelDescription = "ivfpql2 model for testing shared state";

        int dimension = testData.indexData.vectors[0].length;
        SpaceType spaceType = SpaceType.L2;
        int ivfNlist = 4;
        int ivfNprobes = 4;
        int pqCodeSize = 8;
        int pqM = 1;
        int docCount = 100;

        // training data needs to be at least equal to the number of centroids for PQ
        // which is 2^8 = 256. 8 because thats the only valid code_size for HNSWPQ
        int trainingDataCount = 256;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NPROBES, ivfNprobes)
            .field(METHOD_PARAMETER_NLIST, ivfNlist)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_PQ)
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_M, pqM)
            .field(ENCODER_PARAMETER_PQ_CODE_SIZE, pqCodeSize)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        createBasicKnnIndex(trainingIndexName, FIELD_NAME, dimension);
        ingestDataAndTrainModel(modelId, trainingIndexName, FIELD_NAME, dimension, modelDescription, in, trainingDataCount);
        assertTrainingSucceeds(modelId, 360, 1000);

        createIndexFromModelAndIngestDocuments(firstIndexName, modelId, docCount);
        createIndexFromModelAndIngestDocuments(secondIndexName, modelId, docCount);

        doKnnWarmup(List.of(firstIndexName));
        validateSearchWorkflow(firstIndexName, testData.queries, 10);
        doKnnWarmup(List.of(secondIndexName));
        validateSearchWorkflow(secondIndexName, testData.queries, 10);
        deleteKNNIndex(firstIndexName);
        // wait for all index files to be cleaned up from original index. empirically determined to take 25 seconds.
        // will give 15 second buffer from that
        Thread.sleep(1000 * 45);
        validateSearchWorkflow(secondIndexName, testData.queries, 10);
        deleteKNNIndex(secondIndexName);
        deleteModel(modelId);
    }

    @SneakyThrows
    private void createIndexFromModelAndIngestDocuments(String indexName, String modelId, int docCount) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("model_id", modelId)
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();
        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

        for (int i = 0; i < Math.min(testData.indexData.docs.length, docCount); i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                FIELD_NAME,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }
        refreshAllNonSystemIndices();
        assertEquals(Math.min(testData.indexData.docs.length, docCount), getDocCount(indexName));
    }

    @SneakyThrows
    private void validateSearchWorkflow(String indexName, float[][] queries, int k) {
        for (float[] query : queries) {
            Response response = searchKNNIndex(indexName, new KNNQueryBuilder(FIELD_NAME, query, k), k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);
            assertEquals(k, knnResults.size());
        }
    }

    public void testDocUpdate() throws IOException {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";
        Integer dimension = 2;
        SpaceType spaceType = SpaceType.L2;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
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
        SpaceType spaceType = SpaceType.L2;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
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
        int trainingDataCount = 1100;
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

    @SneakyThrows
    public void testByteVectorWithFilter_whenFilterCountGreaterThanK_thenSucceed() {
        String indexName = "test-byte-vector-filter";
        String fieldName = "test-byte-field";
        String categoryField = "category";
        int dimension = 8;
        int k = 5;
        int numDocsPerCategory = 20; // This ensures filterCount (20) > k (5)

        // Create index with byte vector
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field("data_type", "byte")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .startObject(categoryField)
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, builder.toString());

        for (int i = 0; i < numDocsPerCategory; i++) {
            Byte[] vector = new Byte[dimension];
            Arrays.fill(vector, (byte) (i % 128));
            addKnnDocWithAttributes(indexName, "doc_" + i, fieldName, vector, ImmutableMap.of(categoryField, "test_category"));
        }

        refreshIndex(indexName);
        assertEquals(numDocsPerCategory, getDocCount(indexName));

        // Query with filter that matches all documents (filterCount = 20 > k = 5)
        // filterCount * dimension = 20 * 8 = 160 < MAX_DISTANCE_COMPUTATIONS (2048000)
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (byte) 10);

        Response response = searchKNNIndex(
            indexName,
            new KNNQueryBuilder(fieldName, queryVector, k, QueryBuilders.termQuery(categoryField, "test_category")),
            k
        );

        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(responseBody, fieldName);

        // Should return k results even though filter matches more documents
        assertEquals(k, results.size());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testFiltering_whenUsingFaissExactSearchWithIP_thenMatchExpectedScore() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 2)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        final String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);

        final List<Float[]> dataVectors = Arrays.asList(new Float[] { -2.0f, 2.0f }, new Float[] { 2.0f, -2.0f });
        final List<String> ids = Arrays.asList(DOC_ID_1, DOC_ID_2);

        // Ingest all of the documents
        for (int i = 0; i < dataVectors.size(); i++) {
            addKnnDoc(INDEX_NAME, ids.get(i), FIELD_NAME, dataVectors.get(i));
        }
        refreshIndex(INDEX_NAME);

        // Execute the search request with a match all query to ensure exact logic gets called
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 1000));
        float[] queryVector = new float[] { -2.0f, 2.0f };
        int k = 2;
        final Response response = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, queryVector, k, QueryBuilders.matchAllQuery()),
            k
        );
        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<Float> knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);

        // Check that the expected scores are returned
        final List<Float> expectedScores = Arrays.asList(
            KNNEngine.FAISS.score(8.0f, SpaceType.INNER_PRODUCT),
            KNNEngine.FAISS.score(-8.0f, SpaceType.INNER_PRODUCT)
        );
        assertEquals(expectedScores.size(), knnResults.size());
        for (int i = 0; i < expectedScores.size(); i++) {
            assertEquals(expectedScores.get(i), knnResults.get(i), 0.0000001);
        }
    }

    @SneakyThrows
    public void testHNSW_InvalidPQM_thenFail() {
        String trainingIndexName = "training-index";
        String trainingFieldName = "training-field";

        String modelId = "test-model";
        String modelDescription = "test model";

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        int invalidPQM = 3;

        // training data needs to be at least equal to the number of centroids for PQ
        // which is 2^8 = 256. 8 because thats the only valid code_size for HNSWPQ
        int trainingDataCount = 256;

        SpaceType spaceType = SpaceType.L2;

        Integer dimension = testData.indexData.vectors[0].length;

        /*
         * Builds the below json:
         * {
         *   "name": "hnsw",
         *   "engine": "faiss",
         *   "space_type": "l2",
         *   "parameters": {
         *     "encoder": {
         *       "name": "pq",
         *       "parameters": {
         *         "m": 3
         *       }
         *     }
         *   }
         * }
         */

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_PQ)
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_M, invalidPQM)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);

        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        ResponseException re = expectThrows(
            ResponseException.class,
            () -> ingestDataAndTrainModel(modelId, trainingIndexName, trainingFieldName, dimension, modelDescription, in, trainingDataCount)
        );
        assertTrue(
            re.getMessage().contains("Validation Failed: 1: parameter validation failed for MethodComponentContext parameter [encoder].;")
        );
    }

    @SneakyThrows
    public void testIVF_InvalidPQM_thenFail() {
        String trainingIndexName = "training-index";
        String trainingFieldName = "training-field";

        String modelId = "test-model";
        String modelDescription = "test model";

        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        int invalidPQM = 3;

        // training data needs to be at least equal to the number of centroids for PQ
        // which is 2^8 = 256.
        int trainingDataCount = 256;

        int dimension = testData.indexData.vectors[0].length;
        SpaceType spaceType = SpaceType.L2;
        int ivfNlist = 4;
        int ivfNprobes = 4;
        int pqCodeSize = 8;

        /*
         * Builds the below json:
         * {
         *   "name": "ivf",
         *   "engine": "faiss",
         *   "space_type": "l2",
         *   "parameters": {
         *     "nprobes": 8,
         *     "nlist": 4,
         *     "encoder": {
         *       "name": "pq",
         *       "parameters": {
         *         "m": 3,
         *         "code_size": 8
         *       }
         *     }
         *   }
         * }
         */

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NPROBES, ivfNprobes)
            .field(METHOD_PARAMETER_NLIST, ivfNlist)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_PQ)
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_M, invalidPQM)
            .field(ENCODER_PARAMETER_PQ_CODE_SIZE, pqCodeSize)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);

        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        ResponseException re = expectThrows(
            ResponseException.class,
            () -> ingestDataAndTrainModel(modelId, trainingIndexName, trainingFieldName, dimension, modelDescription, in, trainingDataCount)
        );
        assertTrue(
            re.getMessage(),
            re.getMessage().contains("Validation Failed: 1: parameter validation failed for Integer parameter [m].;")
        );
    }

    @SneakyThrows
    public void testIVF_whenBinaryFormat_whenIVF_thenSuccess() {
        String modelId = "test-model-ivf-binary";
        int dimension = 8;

        String trainingIndexName = "train-index-ivf-binary";
        String trainingFieldName = "train-field-ivf-binary";

        String trainIndexMapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(trainingFieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field("data_type", VectorDataType.BINARY.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.HAMMING.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, 24)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 128)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(trainingIndexName, trainIndexMapping);

        int trainingDataCount = 1100;
        bulkIngestRandomBinaryVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        XContentBuilder trainModelXContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, trainingIndexName)
            .field(TRAIN_FIELD_PARAMETER, trainingFieldName)
            .field(DIMENSION, dimension)
            .field(MODEL_DESCRIPTION, "My model description")
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue())
            .field(
                KNN_METHOD,
                Map.of(
                    NAME,
                    METHOD_IVF,
                    KNN_ENGINE,
                    FAISS_NAME,
                    METHOD_PARAMETER_SPACE_TYPE,
                    SpaceType.HAMMING.getValue(),
                    PARAMETERS,
                    Map.of(METHOD_PARAMETER_NLIST, 1, METHOD_PARAMETER_NPROBES, 1)
                )
            )
            .endObject();

        trainModel(modelId, trainModelXContentBuilder);

        // Make sure training succeeds after 30 seconds
        assertTrainingSucceeds(modelId, 30, 1000);

        // Create knn index from model
        String fieldName = "test-field-name-ivf-binary";
        String indexName = "test-index-name-ivf-binary";
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
        Integer[] vector1 = { 11 };
        Integer[] vector2 = { 22 };
        Integer[] vector3 = { 33 };
        Integer[] vector4 = { 44 };
        addKnnDoc(indexName, "1", fieldName, vector1);
        addKnnDoc(indexName, "2", fieldName, vector2);
        addKnnDoc(indexName, "3", fieldName, vector3);
        addKnnDoc(indexName, "4", fieldName, vector4);

        Integer[] queryVector = { 15 };
        int k = 2;

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(fieldName)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Response searchResponse = searchKNNIndex(indexName, queryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(k, results.size());

        deleteKNNIndex(indexName);
        Thread.sleep(45 * 1000);
        deleteModel(modelId);
        deleteKNNIndex(trainingIndexName);
        validateGraphEviction();
    }

    @SneakyThrows
    public void testEndToEnd_whenDoRadiusSearch_whenNoGraphFileIsCreated_whenDistanceThreshold_thenSucceed() {
        final SpaceType spaceType = SpaceType.L2;

        final List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        final List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
        final List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

        final Integer dimension = testData.indexData.vectors[0].length;
        final Settings knnIndexSettings = buildKNNIndexSettings(NEVER_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD);

        // Create an index
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, knnIndexSettings, builder.toString());

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                INDEX_NAME,
                Integer.toString(testData.indexData.docs[i]),
                FIELD_NAME,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        // Assert we have the right number of documents
        refreshAllNonSystemIndices();
        assertEquals(testData.indexData.docs.length, getDocCount(INDEX_NAME));

        final float distance = 300000000000f;
        final List<List<KNNResult>> resultsFromDistance = validateRadiusSearchResults(
            INDEX_NAME,
            FIELD_NAME,
            testData.queries,
            distance,
            null,
            spaceType,
            null,
            null
        );
        assertFalse(resultsFromDistance.isEmpty());
        resultsFromDistance.forEach(result -> { assertFalse(result.isEmpty()); });
        final float score = spaceType.scoreTranslation(distance);
        final List<List<KNNResult>> resultsFromScore = validateRadiusSearchResults(
            INDEX_NAME,
            FIELD_NAME,
            testData.queries,
            null,
            score,
            spaceType,
            null,
            null
        );
        assertFalse(resultsFromScore.isEmpty());
        resultsFromScore.forEach(result -> { assertFalse(result.isEmpty()); });

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testRadialQueryWithFilter_whenNoGraphIsCreated_thenSuccess() {
        setupKNNIndexForFilterQuery(buildKNNIndexSettings(NEVER_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD));

        final float[][] searchVector = new float[][] { { 3.3f, 3.0f, 5.0f } };
        TermQueryBuilder termQueryBuilder = QueryBuilders.termQuery("color", "red");
        List<String> expectedDocIds = Arrays.asList(DOC_ID_3);

        float distance = 15f;
        List<List<KNNResult>> queryResult = validateRadiusSearchResults(
            INDEX_NAME,
            FIELD_NAME,
            searchVector,
            distance,
            null,
            SpaceType.L2,
            termQueryBuilder,
            null
        );

        assertEquals(1, queryResult.get(0).size());
        assertEquals(expectedDocIds.get(0), queryResult.get(0).get(0).getDocId());

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testEndToEnd_whenDoRadiusSearch_withByteVector_thenSucceed() {
        SpaceType spaceType = SpaceType.L2;
        String indexName = "test-index-byte-radial";
        String fieldName = "test-field-byte-radial";
        int dimension = 8;

        // Create an index with byte vector data type
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field("data_type", VectorDataType.BYTE.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, 16)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 128)
            .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, 128)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(indexName, mapping);

        // Index byte vectors with known distances
        Byte[] vector1 = { 1, 2, 3, 4, 5, 6, 7, 8 };        // Close to query
        Byte[] vector2 = { 10, 20, 30, 40, 50, 60, 70, 80 }; // Medium distance
        Byte[] vector3 = { 100, 110, 120, 127, 126, 125, 124, 123 }; // Far from query

        addKnnDoc(indexName, "1", fieldName, vector1);
        addKnnDoc(indexName, "2", fieldName, vector2);
        addKnnDoc(indexName, "3", fieldName, vector3);

        refreshAllNonSystemIndices();
        assertEquals(3, getDocCount(indexName));

        float[] queryVector = { 5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 35.0f, 40.0f };

        // Calculate expected L2 squared distances:
        // vector1: (5-1)² + (10-2)² + (15-3)² + (20-4)² + (25-5)² + (30-6)² + (35-7)² + (40-8)² = 16 + 64 + 144 + 256 + 400 + 576 + 784 +
        // 1024 = 3264
        // vector2: (5-10)² + (10-20)² + (15-30)² + (20-40)² + (25-50)² + (30-60)² + (35-70)² + (40-80)² = 25 + 100 + 225 + 400 + 625 + 900
        // + 1225 + 1600 = 5100
        // vector3: Much larger distance due to large values

        // Test 1: Small distance threshold - should only return vector1
        float smallDistance = 4000f; // Only vector1 should match (distance² = 3264)
        List<List<KNNResult>> results1 = validateRadiusSearchResults(
            indexName,
            fieldName,
            new float[][] { queryVector },
            smallDistance,
            null,
            spaceType,
            null,
            null
        );
        assertEquals("Should return exactly 1 result for small distance", 1, results1.get(0).size());
        assertEquals("Should return document 1", "1", results1.get(0).get(0).getDocId());

        // Test 2: Medium distance threshold - should return vector1 and vector2
        float mediumDistance = 6000f; // vector1 and vector2 should match
        List<List<KNNResult>> results2 = validateRadiusSearchResults(
            indexName,
            fieldName,
            new float[][] { queryVector },
            mediumDistance,
            null,
            spaceType,
            null,
            null
        );
        assertEquals("Should return exactly 2 results for medium distance", 2, results2.get(0).size());

        // Verify the returned documents are sorted by distance (closest first)
        List<String> docIds = results2.get(0).stream().map(KNNResult::getDocId).collect(Collectors.toList());
        assertTrue("Should contain document 1", docIds.contains("1"));
        assertTrue("Should contain document 2", docIds.contains("2"));

        // Test 3: Large distance threshold - should return all vectors
        float largeDistance = 500000f;
        List<List<KNNResult>> results3 = validateRadiusSearchResults(
            indexName,
            fieldName,
            new float[][] { queryVector },
            largeDistance,
            null,
            spaceType,
            null,
            null
        );
        assertEquals("Should return all 3 results for large distance", 3, results3.get(0).size());

        // Test 4: Very small distance threshold - should return no results
        float tinyDistance = 1000f; // No vectors should match
        List<List<KNNResult>> results4 = validateRadiusSearchResults(
            indexName,
            fieldName,
            new float[][] { queryVector },
            tinyDistance,
            null,
            spaceType,
            null,
            null
        );
        assertEquals("Should return no results for very small distance", 0, results4.get(0).size());

        // Delete index
        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testQueryWithFilter_whenNonExistingFieldUsedInFilter_thenSuccessful() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, VECTOR_DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .startObject(INTEGER_FIELD_NAME)
            .field(TYPE_FIELD_NAME, FILED_TYPE_INTEGER)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(INDEX_NAME, mapping);

        Float[] vector = new Float[] { 2.0f, 4.5f, 6.5f };

        String documentAsString = XContentFactory.jsonBuilder()
            .startObject()
            .field(INTEGER_FIELD_NAME, 5)
            .field(FIELD_NAME, vector)
            .endObject()
            .toString();

        addKnnDoc(INDEX_NAME, DOC_ID_1, documentAsString);

        refreshIndex(INDEX_NAME);
        assertEquals(1, getDocCount(INDEX_NAME));

        float[] searchVector = new float[] { 1.0f, 2.1f, 3.9f };
        int k = 10;

        // use filter where nonexistent field is must, we should have no results
        QueryBuilder filterWithRequiredNonExistentField = QueryBuilders.boolQuery()
            .must(QueryBuilders.rangeQuery(NON_EXISTENT_INTEGER_FIELD_NAME).gte(1));
        Response searchWithRequiredNonExistentFiledInFilterResponse = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, k, filterWithRequiredNonExistentField),
            k
        );
        List<KNNResult> resultsQuery1 = parseSearchResponse(
            EntityUtils.toString(searchWithRequiredNonExistentFiledInFilterResponse.getEntity()),
            FIELD_NAME
        );
        assertTrue(resultsQuery1.isEmpty());

        // use filter with non existent field as optional, we should have some results
        QueryBuilder filterWithOptionalNonExistentField = QueryBuilders.boolQuery()
            .should(QueryBuilders.rangeQuery(NON_EXISTENT_INTEGER_FIELD_NAME).gte(1))
            .must(QueryBuilders.rangeQuery(INTEGER_FIELD_NAME).gte(1));
        Response searchWithOptionalNonExistentFiledInFilterResponse = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, k, filterWithOptionalNonExistentField),
            k
        );
        List<KNNResult> resultsQuery2 = parseSearchResponse(
            EntityUtils.toString(searchWithOptionalNonExistentFiledInFilterResponse.getEntity()),
            FIELD_NAME
        );
        assertEquals(1, resultsQuery2.size());
    }

    public void testCosineSimilarity_withHNSW_withExactSearch_thenSucceed() throws Exception {
        testCosineSimilarityForApproximateSearch(NEVER_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD);
    }

    @ExpectRemoteBuildValidation
    public void testCosineSimilarity_withHNSW_withApproximate_thenSucceed() throws Exception {
        testCosineSimilarityForApproximateSearch(ALWAYS_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD);
        validateGraphEviction();
    }

    @ExpectRemoteBuildValidation
    public void testCosineSimilarity_withGraph_withRadialSearch_withDistanceThreshold_thenSucceed() throws Exception {
        testCosineSimilarityForRadialSearch(ALWAYS_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD, null, 0.1f);
        validateGraphEviction();
    }

    @ExpectRemoteBuildValidation
    public void testCosineSimilarity_withGraph_withRadialSearch_withScore_thenSucceed() throws Exception {
        testCosineSimilarityForRadialSearch(ALWAYS_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD, 0.9f, null);
        validateGraphEviction();
    }

    public void testCosineSimilarity_withNoGraphs_withRadialSearch_withDistanceThreshold_thenSucceed() throws Exception {
        testCosineSimilarityForRadialSearch(NEVER_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD, null, 0.1f);
        validateGraphEviction();
    }

    public void testCosineSimilarity_withNoGraphs_withRadialSearch_withScore_thenSucceed() throws Exception {
        testCosineSimilarityForRadialSearch(NEVER_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD, 0.9f, null);
        validateGraphEviction();
    }

    public void testEndToEnd_withApproxAndExactSearch_inSameIndex_ForCosineSpaceType() throws Exception {
        String indexName = randomLowerCaseString();
        String fieldName = randomLowerCaseString();
        SpaceType spaceType = SpaceType.COSINESIMIL;
        Integer dimension = testData.indexData.vectors[0].length;

        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(indexName, buildKNNIndexSettings(0), mapping);

        // Index one document
        addKnnDoc(indexName, randomAlphaOfLength(5), fieldName, Floats.asList(testData.indexData.vectors[0]).toArray());

        // Assert we have the right number of documents in the index
        refreshAllIndices();
        assertEquals(1, getDocCount(indexName));
        // update threshold setting to skip building graph
        updateIndexSettings(indexName, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, -1));
        // add duplicate document with different id
        addKnnDoc(indexName, randomAlphaOfLength(5), fieldName, Floats.asList(testData.indexData.vectors[0]).toArray());
        assertEquals(2, getDocCount(indexName));
        final int k = 2;
        // search index
        Response response = searchKNNIndex(
            indexName,
            KNNQueryBuilder.builder().fieldName(fieldName).vector(testData.queries[0]).k(k).build(),
            k
        );
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
        assertEquals(k, knnResults.size());

        List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);

        // both document should have identical score
        assertEquals(actualScores.get(0), actualScores.get(1), 0.001);
    }

    protected void setupKNNIndexForFilterQuery() throws Exception {
        setupKNNIndexForFilterQuery(getKNNDefaultIndexSettings());
    }

    protected void setupKNNIndexForFilterQuery(Settings settings) throws Exception {
        // Create Mappings
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", 3)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        final String mapping = builder.toString();

        createKnnIndex(INDEX_NAME, settings, mapping);

        addKnnDocWithAttributes(
            DOC_ID_1,
            new float[] { 6.0f, 7.9f, 3.1f },
            ImmutableMap.of(COLOR_FIELD_NAME, "red", TASTE_FIELD_NAME, "sweet")
        );
        addKnnDocWithAttributes(DOC_ID_2, new float[] { 3.2f, 2.1f, 4.8f }, ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes(DOC_ID_3, new float[] { 4.1f, 5.0f, 7.1f }, ImmutableMap.of(COLOR_FIELD_NAME, "red"));

        refreshIndex(INDEX_NAME);
    }

    @SneakyThrows
    private void queryTestData(final String indexName, final String fieldName, final int dimension, final int numDocs) {
        queryTestData(indexName, fieldName, dimension, numDocs, null);
    }

    private void queryTestData(
        final String indexName,
        final String fieldName,
        final int dimension,
        final int numDocs,
        Map<String, ?> methodParams
    ) throws IOException, ParseException {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);
        int k = 10;

        Response searchResponse = searchKNNIndex(indexName, buildSearchQuery(fieldName, k, queryVector, methodParams), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
        }
    }

    private void indexTestData(final String indexName, final String fieldName, final int dimension, final int numDocs) throws Exception {
        for (int i = 0; i < numDocs; i++) {
            float[] indexVector = new float[dimension];
            Arrays.fill(indexVector, (float) i);
            addKnnDocWithAttributes(indexName, Integer.toString(i), fieldName, indexVector, ImmutableMap.of("rating", String.valueOf(i)));
        }

        // Assert that all docs are ingested
        refreshAllNonSystemIndices();
        assertEquals(numDocs, getDocCount(indexName));
    }

    private void validateGraphEviction() throws Exception {
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

    private List<List<KNNResult>> validateRadiusSearchResults(
        String indexName,
        String fieldName,
        float[][] queryVectors,
        Float distanceThreshold,
        Float scoreThreshold,
        final SpaceType spaceType,
        TermQueryBuilder filterQuery,
        Map<String, ?> methodParameters
    ) throws IOException, ParseException {
        List<List<KNNResult>> queryResults = new ArrayList<>();
        for (float[] queryVector : queryVectors) {
            XContentBuilder queryBuilder = XContentFactory.jsonBuilder().startObject().startObject("query");
            queryBuilder.startObject("knn");
            queryBuilder.startObject(fieldName);
            queryBuilder.field("vector", queryVector);
            if (distanceThreshold != null) {
                queryBuilder.field(MAX_DISTANCE, distanceThreshold);
            } else if (scoreThreshold != null) {
                queryBuilder.field(MIN_SCORE, scoreThreshold);
            } else {
                throw new IllegalArgumentException("Invalid threshold");
            }
            if (filterQuery != null) {
                queryBuilder.field("filter", filterQuery);
            }
            if (methodParameters != null && methodParameters.size() > 0) {
                queryBuilder.startObject(METHOD_PARAMETER);
                for (Map.Entry<String, ?> entry : methodParameters.entrySet()) {
                    queryBuilder.field(entry.getKey(), entry.getValue());
                }
                queryBuilder.endObject();
            }
            queryBuilder.endObject();
            queryBuilder.endObject();
            queryBuilder.endObject().endObject();
            final String responseBody = EntityUtils.toString(searchKNNIndex(indexName, queryBuilder, 10).getEntity());

            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);

            for (KNNResult knnResult : knnResults) {
                float[] vector = knnResult.getVector();
                float distance = TestUtils.computeDistFromSpaceType(spaceType, vector, queryVector);
                if (spaceType == SpaceType.L2) {
                    assertTrue(KNNScoringUtil.l2Squared(queryVector, vector) <= distance);
                } else if (spaceType == SpaceType.INNER_PRODUCT) {
                    assertTrue(KNNScoringUtil.innerProduct(queryVector, vector) >= distance);
                } else if (spaceType == SpaceType.COSINESIMIL) {
                    assertTrue(KNNScoringUtil.cosinesimil(queryVector, vector) >= distance);
                } else {
                    throw new IllegalArgumentException("Invalid space type");
                }
            }
            queryResults.add(knnResults);
        }
        return queryResults;
    }

    private void testCosineSimilarityForApproximateSearch(int approximateThreshold) throws Exception {
        String indexName = randomLowerCaseString();
        String fieldName = randomLowerCaseString();
        SpaceType spaceType = SpaceType.COSINESIMIL;
        indexTestData(approximateThreshold, indexName, spaceType, fieldName);

        // search index
        validateNearestNeighborsSearch(indexName, fieldName, spaceType, 10, VectorUtil::cosine);

        // Delete index
        deleteKNNIndex(indexName);
    }

    private void testCosineSimilarityForRadialSearch(int approximateThreshold, Float score, Float distance) throws Exception {
        String indexName = randomLowerCaseString();
        String fieldName = randomLowerCaseString();
        SpaceType spaceType = SpaceType.COSINESIMIL;
        indexTestData(approximateThreshold, indexName, spaceType, fieldName);

        // search index
        validateRadiusSearchResults(indexName, fieldName, testData.queries, distance, score, spaceType, null, null);

        // Delete index
        deleteKNNIndex(indexName);
    }

    private void indexTestData(int approximateThreshold, String indexName, SpaceType spaceType, String fieldName) throws Exception {
        Integer dimension = testData.indexData.vectors[0].length;
        // Create an index
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
            .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(indexName, buildKNNIndexSettings(approximateThreshold), mapping);

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                fieldName,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }

        refreshAllIndices();
        // Assert we have the right number of documents in the index
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));
    }

    @SneakyThrows
    private void validateNearestNeighborsSearch(
        final String indexName,
        final String fieldName,
        final SpaceType spaceType,
        final int k,
        final BiFunction<float[], float[], Float> scoringFunction
    ) {
        for (int i = 0; i < testData.queries.length; i++) {
            final Response response = searchKNNIndex(
                indexName,
                KNNQueryBuilder.builder().fieldName(fieldName).vector(testData.queries[i]).k(k).build(),
                k
            );
            final String responseBody = EntityUtils.toString(response.getEntity());
            final List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            final List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                final float[] primitiveArray = knnResults.get(j).getVector();
                assertEquals(
                    KNNEngine.FAISS.score(scoringFunction.apply(testData.queries[i], primitiveArray), spaceType),
                    actualScores.get(j),
                    0.0001
                );
            }
        }
    }

}
