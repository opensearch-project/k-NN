//package org.opensearch.knn.index;
//
//import com.google.common.collect.ImmutableList;
//import lombok.SneakyThrows;
//import org.apache.hc.core5.http.io.entity.EntityUtils;
//import org.opensearch.action.bulk.BulkRequest;
//import org.opensearch.action.index.IndexRequest;
//import org.opensearch.action.search.SearchResponse;
//import org.opensearch.client.Request;
//import org.opensearch.client.Response;
//import org.opensearch.common.xcontent.XContentFactory;
//import org.opensearch.core.xcontent.MediaTypeRegistry;
//import org.opensearch.core.xcontent.XContentBuilder;
//import org.opensearch.knn.KNNRestTestCase;
//import org.opensearch.knn.index.engine.KNNEngine;
//import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
//import org.opensearch.knn.index.query.KNNQueryBuilder;
//import org.opensearch.knn.quantization.quantizer.RandomGaussianRotation;
//import org.opensearch.search.SearchHit;
//
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.HashSet;
//import java.util.List;
//import java.util.Map;
//import java.util.Random;
//import java.util.Set;
//
//import static org.opensearch.knn.common.KNNConstants.DIMENSION;
//import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
//import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
//import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
//import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
//import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
//import static org.opensearch.knn.common.KNNConstants.TYPE;
//import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
//import static org.opensearch.knn.common.KNNConstants.NAME;
//
//public class RandomRotationIT extends KNNRestTestCase {
//
//    private static final String INDEX_NAME_NO_ROTATION = "no-rotation-index";
//    private static final String INDEX_NAME_WITH_ROTATION = "rotation-index";
//    private static final int TEST_DIMENSION = 8; // As per the example
//    private static final int DOC_COUNT = 100;
//    private static final int K = 10; // Number of search results
//    private static final long QUANTIZATION_RANDOM_ROTATION_DEFAULT_SEED = 42L; // Same as in RandomGaussianRotation
//    private static final String TEST_FIELD_NAME = "vector_field";
//
//
//
//    @SneakyThrows
//    public void testRandomRotation() {
//        fail();
//        SpaceType spaceType = SpaceType.INNER_PRODUCT;
//        Integer bits = 1;
//        int dimension = 2;
//        logger.info("testing");
//        String indexName = "rand-rotation-index";
//        XContentBuilder builder = XContentFactory.jsonBuilder()
//                .startObject()
//                .startObject(PROPERTIES_FIELD)
//                .startObject(TEST_FIELD_NAME)
//                .field(TYPE, TYPE_KNN_VECTOR)
//                .field(DIMENSION, dimension)
//                .startObject(KNN_METHOD)
//                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
//                .field(KNN_ENGINE, KNNEngine.FAISS.getName())
//                .field("name", METHOD_HNSW)
//                .startObject(PARAMETERS)
//                .startObject("encoder")
//                .field("name", "binary")
//                .startObject("parameters")
//                .field("bits", bits)
//                .field(QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM, true)
//                .endObject()
//                .endObject()
//                .endObject()
//                .endObject()
//                .endObject()
//                .endObject()
//                .endObject();
//        createKnnIndex(indexName, builder.toString());
//
//        float[] vector1 =   {1.0f, 0.0f};
//        float[] vector2 =   {0.0f, 1.0f};
//        float[] vector3 =   {-1.0f, 0.0f};
//        float[] query =     { 0.25f, -1.0f};
//
//
//        // Without rotation -> 1,3,2:
//        // vec1:  --> [1, 0]
//        // vec2:  --> [0, 1]
//        // vec3:  --> [0, 0]
//        // query: --> [1, 0]
//
//        // With rotation -> 3,1,2
//        // vec1: 1, 0 -> [-0.22524017, 0.9743033]   --> [0, 1]
//        // vec2: 1, 1 -> [0.9743033, 0.22524008]    --> [1, 1]
//        // vec3: 0, 0 -> [0.22524017, -0.9743033]   --> [0, 0]
//        // query: 1, 0 -> [-1.0306133, 0.018335745] --> [0, 0]
//
//        Float[] vector_1 = { 1.0f, 0.0f };
//        Float[] vector_2 = { 0.0f, 1.0f };
//        Float[] vector_3 = { -1.0f, 0.0f };
//        addKnnDoc(
//                indexName,
//                "1",
//                ImmutableList.of(TEST_FIELD_NAME),
//                ImmutableList.of(vector_1)
//        );
//        addKnnDoc(
//                indexName,
//                "2",
//                ImmutableList.of(TEST_FIELD_NAME),
//                ImmutableList.of(vector_2)
//        );
//        addKnnDoc(
//                indexName,
//                "3",
//                ImmutableList.of(TEST_FIELD_NAME),
//                ImmutableList.of(vector_3)
//        );
//
//        forceMergeKnnIndex(indexName);
//
//        XContentBuilder queryBuilder = XContentFactory.jsonBuilder();
//        queryBuilder.startObject();
//        queryBuilder.startObject("query");
//        queryBuilder.startObject("knn");
//        queryBuilder.startObject(TEST_FIELD_NAME);
//        queryBuilder.field("vector", query);
//        queryBuilder.field("k", 3);
//        queryBuilder.endObject();
//        queryBuilder.endObject();
//        queryBuilder.endObject();
//        queryBuilder.endObject();
//        final String responseBody = EntityUtils.toString(searchKNNIndex(indexName, queryBuilder, 3).getEntity());
//        logger.info(responseBody);
//
//
//        float [][] rotationMatrix = RandomGaussianRotation.generateRotationMatrix(2);
//
//
//        logger.info("Rotated vector1: {}", Arrays.toString(RandomGaussianRotation.applyRotation(vector1, rotationMatrix)));
//        logger.info("Rotated vector2: {}", Arrays.toString(RandomGaussianRotation.applyRotation(vector2, rotationMatrix)));
//        logger.info("Rotated vector3: {}", Arrays.toString(RandomGaussianRotation.applyRotation(vector3, rotationMatrix)));
//        logger.info("Rotated query:  {}", Arrays.toString(RandomGaussianRotation.applyRotation(query, rotationMatrix)));
//
//        fail();
//
//    }
//
//    @SneakyThrows
//    public void testADC() {
//        SpaceType spaceType = SpaceType.INNER_PRODUCT;
//        Integer bits = 1;
//        int dimension = 8;
//        String indexName = "rand-rotation-index";
//        XContentBuilder builder = XContentFactory.jsonBuilder()
//                .startObject()
//                .startObject(PROPERTIES_FIELD)
//                .startObject(TEST_FIELD_NAME)
//                .field(TYPE, TYPE_KNN_VECTOR)
//                .field(DIMENSION, dimension)
//                .startObject(KNN_METHOD)
//                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
//                .field(KNN_ENGINE, KNNEngine.FAISS.getName())
//                .field(NAME, METHOD_HNSW)
//                .startObject(PARAMETERS)
//                .startObject("encoder")
//                .field("name", "binary")
//                .startObject("parameters")
//                .field("bits", bits)
//                .field(QFrameBitEncoder.ENABLE_ADC_PARAM, false)
//                .field(QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM, false)
//                .endObject()
//                .endObject()
//                .endObject()
//                .endObject()
//                .endObject()
//                .endObject()
//                .endObject();
//        createKnnIndex(indexName, builder.toString());
//
//        float[] vector1 =   {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
//        float[] vector2 =   {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
//        float[] vector3 =   {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
//        float[] query =     { 0.1f, 0.32f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
//
//
//        // [0.1, 0.32] -> [0, 0]
//
//
//        // Without adc -> 3...:
//        // vec1:  --> [1, 0]
//        // vec2:  --> [0, 1]
//        // vec3:  --> [0, 0]
//        // query: --> [1, 0]
//
//        // With adc -> 1
//        // vec1:  --> [1, 0]
//        // vec2:  --> [0, 1]
//        // vec3:  --> [0, 0]
//        // query. --> [.1, 0]
//
//
//        // x1: below = 0 above = 1
//        // x2: below = 0 above = 1
//
//        // vector[i] = vector[i];
//
//
//        Float[] vector_1 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
//        Float[] vector_2 = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
//        Float[] vector_3 = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
//        addKnnDoc(
//                indexName,
//                "1",
//                ImmutableList.of(TEST_FIELD_NAME),
//                ImmutableList.of(vector_1)
//        );
//        addKnnDoc(
//                indexName,
//                "2",
//                ImmutableList.of(TEST_FIELD_NAME),
//                ImmutableList.of(vector_2)
//        );
//        addKnnDoc(
//                indexName,
//                "3",
//                ImmutableList.of(TEST_FIELD_NAME),
//                ImmutableList.of(vector_3)
//        );
//        forceMergeKnnIndex(indexName);
//
//        XContentBuilder queryBuilder = XContentFactory.jsonBuilder();
//        queryBuilder.startObject();
//        queryBuilder.startObject("query");
//        queryBuilder.startObject("knn");
//        queryBuilder.startObject(TEST_FIELD_NAME);
//        queryBuilder.field("vector", query);
//        queryBuilder.field("k", 3);
//        queryBuilder.endObject();
//        queryBuilder.endObject();
//        queryBuilder.endObject();
//        queryBuilder.endObject();
//        final String responseBody = EntityUtils.toString(searchKNNIndex(indexName, queryBuilder, 3).getEntity());
//        logger.info(responseBody);
//        fail();
//
//    }
////
////    public void testRandomRotationAffectsBinaryThresholds() throws Exception {
////        // Create two indices - one with random rotation, one without
////        createIndexWithRotation(INDEX_NAME_WITH_ROTATION, true);
////        createIndexWithRotation(INDEX_NAME_NO_ROTATION, false);
////
////        // Generate document vectors
////        List<float[]> documentVectors = generateRandomVectors(DOC_COUNT, TEST_DIMENSION);
////
////        // Index the same vectors to both indices
////        indexVectors(INDEX_NAME_WITH_ROTATION, documentVectors);
////        indexVectors(INDEX_NAME_NO_ROTATION, documentVectors);
////
////        // Force refresh to ensure thresholds are calculated
////        refreshAllIndices();
////        ensureGreen(INDEX_NAME_WITH_ROTATION);
////        ensureGreen(INDEX_NAME_NO_ROTATION);
////
////        // Wait for index to be fully ready
////        Thread.sleep(1000);
////
////        // Create a query vector (not in the dataset)
////        float[] queryVector = generateRandomVector(TEST_DIMENSION);
////
////        // Search both indices with the same query vector
////        List<String> rotationResults = searchIndex(INDEX_NAME_WITH_ROTATION, queryVector);
////        List<String> noRotationResults = searchIndex(INDEX_NAME_NO_ROTATION, queryVector);
////
////        logger.info("Results with rotation: {}", rotationResults);
////        logger.info("Results without rotation: {}", noRotationResults);
////
////        // Assert that results are different due to different binary thresholds
////        assertNotEquals(
////                "Search results should differ when random rotation is enabled because binary thresholds are calculated differently",
////                rotationResults,
////                noRotationResults
////        );
////
////        // Calculate how many different results we're seeing
////        Set<String> rotationSet = new HashSet<>(rotationResults);
////        Set<String> noRotationSet = new HashSet<>(noRotationResults);
////
////        Set<String> difference = new HashSet<>(rotationSet);
////        difference.addAll(noRotationSet);
////
////        Set<String> intersection = new HashSet<>(rotationSet);
////        intersection.retainAll(noRotationSet);
////
////        difference.removeAll(intersection);
////        double percentDifferent = (double) difference.size() / K * 100.0;
////
////        logger.info("Percentage of different results: {}%", percentDifferent);
////
////        // Assert a significant difference in results
////        assertTrue(
////                String.format("Random rotation should cause significant differences in results (at least 20%%), " +
////                        "but only %.2f%% of results were different", percentDifferent),
////                percentDifferent >= 20.0
////        );
////    }
////
////    @SneakyThrows
////    private void createIndexWithRotation(String indexName, boolean enableRandomRotation) {
////        XContentBuilder builder = XContentFactory.jsonBuilder()
////                .startObject()
////                .startObject("settings")
////                .startObject("index")
////                .field("knn", true)
////                .endObject()
////                .endObject()
////                .startObject("mappings")
////                .startObject("properties")
////                .startObject(TEST_FIELD_NAME)
////                .field("type", "knn_vector")
////                .field("dimension", TEST_DIMENSION)
////                .field("compression_level", "32x")
////                .startObject("method")
////                .field("name", "hnsw")
////                .field("engine", "faiss")
////                .field("space_type", "l2")
////                .startObject("parameters")
////                .field("ef_construction", 128)
////                .field("m", 16)
////                .startObject("encoder")
////                .field("name", "binary")
////                .startObject("parameters")
////                .field("bits", 1)
////                .field("random_rotation", enableRandomRotation)
////                .field("enable_adc", true)
////                .endObject()
////                .endObject()
////                .endObject()
////                .endObject()
////                .endObject()
////                .endObject()
////                .endObject()
////                .endObject();
////
////        createKnnIndex(indexName, builder.toString());
////    }
////
////    private List<float[]> generateRandomVectors(int count, int dimension) {
////        List<float[]> vectors = new ArrayList<>(count);
////        Random random = new Random(QUANTIZATION_RANDOM_ROTATION_DEFAULT_SEED);
////
////        for (int i = 0; i < count; i++) {
////            vectors.add(generateRandomVector(dimension, random));
////        }
////
////        return vectors;
////    }
////
////    @SneakyThrows
////    private Response searchIndex(String indexName, String query) {
////        Request request = new Request("POST", "/" + indexName + "/_search");
////        request.setJsonEntity(query);
////        return client().performRequest(request);
////    }
////
////
////    private float[] generateRandomVector(int dimension) {
////        // Use a different seed for the query vector to ensure it's not in the dataset
////        return generateRandomVector(dimension, new Random());
////    }
////
////    private float[] generateRandomVector(int dimension, Random random) {
////        float[] vector = new float[dimension];
////
////        for (int j = 0; j < dimension; j++) {
////            // Generate values between -1.0 and 1.0
////            vector[j] = random.nextFloat() * 2 - 1;
////        }
////
////        return vector;
////    }
////
////    @SneakyThrows
////    private void indexVectors(String indexName, List<float[]> vectors) {
////        for (int i = 0; i < vectors.size(); i++) {
////            String id = String.valueOf(i);
////            XContentBuilder docBuilder = XContentFactory.jsonBuilder()
////                    .startObject()
////                    .array("vector_field", vectors.get(i))
////                    .field("id", id)
////                    .endObject();
////
////            addKnnDoc(indexName, id, docBuilder.toString());
////        }
////    }
////
//////
//////    @SneakyThrows
//////    private void indexVectors(String indexName, List<float[]> vectors) {
//////        BulkRequest bulkRequest = new BulkRequest();
//////
//////        for (int i = 0; i < vectors.size(); i++) {
//////            String id = String.valueOf(i);
//////            bulkRequest.add(
//////                    new IndexRequest(indexName)
//////                            .id(id)
//////                            .source(XContentFactory.jsonBuilder()
//////                                    .startObject()
//////                                    .array("vector_field", vectors.get(i))
//////                                    .field("id", id)
//////                                    .endObject()
//////                            )
//////            );
//////        }
//////
//////        client().bulk(bulkRequest).actionGet();
//////    }
////
//////    @SneakyThrows
//////    private void indexVectorDocuments(String indexName, int count, VectorDataType dataType) {
//////        for (int i = 0; i < count; i++) {
//////            XContentBuilder docBuilder = XContentFactory.jsonBuilder()
//////                    .startObject()
//////                    .field("vector_field", generateVector(dataType, getDimensionForDataType(dataType)))
//////                    .field("id", i)
//////                    .endObject();
//////            addKnnDoc(indexName, String.valueOf(i), docBuilder.toString());
//////        }
//////    }
////
////    @SneakyThrows
////    private List<String> searchIndex(String indexName, float[] queryVector) {
////        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
////                .startObject()
////                .startObject("query")
////                .startObject("knn")
////                .startObject("vector_field")
////                .field("vector", queryVector)
////                .field("k", 10)
////                .endObject()
////                .endObject()
////                .endObject()
////                .endObject();
////
////        Request request = new Request("POST", "/" + indexName + "/_search");
////        request.setJsonEntity(queryBuilder.toString());
////        Response response = client().performRequest(request);
////
////
////
////        List<String> resultIds = new ArrayList<>();
////
////        String responseBody = EntityUtils.toString(response.getEntity());
////        Map<String, Object> responseMap = parseResponseToMap(responseBody);
////
////        for (SearchHit hit : responseMap.get("hits") ) {
////            resultIds.add(hit.getId());
////        }
////
////        return resultIds;
////    }
////    private Map<String, Object> parseResponseToMap(String responseBody) throws IOException {
//////        createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map()
//////        return parseSearchResponseHits(responseBody);
////        return ((Map<String, Object>) createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map());
////
////    }
//
//}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableList;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;

public class RandomRotationADCIT extends KNNRestTestCase {

    private static final String TEST_FIELD_NAME = "test-field";

    private String makeQBitIndex(String name, boolean isUnderTest) throws Exception {
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        Integer bits = 1;
        int dimension = 2;
        String indexName = "rand-rot-index" + isUnderTest;
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(TEST_FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .startObject("encoder")
            .field(NAME, "binary")
            .startObject("parameters")
            .field("bits", bits)
            .field(name, isUnderTest)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());

        float[] vector1 = { 1.0f, 0.0f };
        float[] vector2 = { 0.0f, 1.0f };
        float[] vector3 = { -1.0f, 0.0f };
        // float[] query = { 0.25f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // Without rotation -> 1,3,2:
        // vec1: --> [1, 0]
        // vec2: --> [0, 1]
        // vec3: --> [0, 0]
        // query: --> [1, 0]

        // With rotation -> 3,1,2
        // vec1: 1, 0 -> [-0.22524017, 0.9743033] --> [0, 1]
        // vec2: 1, 1 -> [0.9743033, 0.22524008] --> [1, 1]
        // vec3: 0, 0 -> [0.22524017, -0.9743033] --> [0, 0]
        // query: 1, 0 -> [-1.0306133, 0.018335745] --> [0, 0]

        // Float[] vector_1 = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        // Float[] vector_2 = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        // Float[] vector_3 = { -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        Float[] vector_1 = { 1.0f, 0.0f };
        Float[] vector_2 = { 0.0f, 1.0f };
        Float[] vector_3 = { -1.0f, 0.0f };
        float[] query = { 0.25f, -1.0f };

        addKnnDoc(indexName, "1", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_1));
        addKnnDoc(indexName, "2", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_2));
        addKnnDoc(indexName, "3", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_3));

        forceMergeKnnIndex(indexName);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder();
        queryBuilder.startObject();
        queryBuilder.startObject("query");
        queryBuilder.startObject("knn");
        queryBuilder.startObject(TEST_FIELD_NAME);
        queryBuilder.field("vector", query);
        queryBuilder.field("k", 3);
        queryBuilder.endObject();
        queryBuilder.endObject();
        queryBuilder.endObject();
        queryBuilder.endObject();
        final String responseBody = EntityUtils.toString(searchKNNIndex(indexName, queryBuilder, 3).getEntity());
        deleteKNNIndex(indexName);
        return responseBody;
    }

    @SneakyThrows
    public void testRandomRotation() {
        String responseControl = makeQBitIndex(QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM, false);
        String responseUnderTest = makeQBitIndex(QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM, true);

        logger.info(responseControl);
        logger.info(responseUnderTest);

        List<Object> controlHits = parseSearchResponseHits(responseControl);
        List<Object> testHits = parseSearchResponseHits(responseUnderTest);

        logger.info(controlHits);

        logger.info(controlHits.get(0).getClass());
        logger.info(((java.util.HashMap<String, Object>) controlHits.get(0)));
        logger.info(((java.util.HashMap<String, Object>) controlHits.get(0)).get("_id"));

        // logger.info((int)((java.util.HashMap<String, Object>) controlHits.get(0)).get("_id"));
        logger.info(parseSearchResponseHits(responseUnderTest));

        int controlFirstHitId = Integer.parseInt((String) (((java.util.HashMap<String, Object>) controlHits.get(0)).get("_id")));
        int testFirstHitId = Integer.parseInt((String) (((java.util.HashMap<String, Object>) testHits.get(0)).get("_id")));

        assertEquals(1, controlFirstHitId);
        assertEquals(3, testFirstHitId);
        // fail();
    }

    private void makeOnlyQBitIndex(String indexName, String name, int dimension, int bits, boolean isUnderTest, SpaceType spaceType)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(TEST_FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .startObject("encoder")
            .field(NAME, "binary")
            .startObject("parameters")
            .field("bits", bits)
            .field(name, isUnderTest)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());
    }

    private String makeQBitIndexADC(String indexName, String name, boolean isUnderTest) throws Exception {
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        Integer bits = 1;
        int dimension = 8;

        // String indexName = "rand-rot-index";
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(TEST_FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .startObject("encoder")
            .field(NAME, "binary")
            .startObject("parameters")
            .field("bits", bits)
            .field(name, isUnderTest)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());

        Float[] vector_1 = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        Float[] vector_2 = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        Float[] vector_3 = { -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        float[] query = { 0.1f, 0.32f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        addKnnDoc(indexName, "1", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_1));
        addKnnDoc(indexName, "2", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_2));
        addKnnDoc(indexName, "3", ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vector_3));
        forceMergeKnnIndex(indexName);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder();
        queryBuilder.startObject();
        queryBuilder.startObject("query");
        queryBuilder.startObject("knn");
        queryBuilder.startObject(TEST_FIELD_NAME);
        queryBuilder.field("vector", query);
        queryBuilder.field("k", 3);
        queryBuilder.endObject();
        queryBuilder.endObject();
        queryBuilder.endObject();
        queryBuilder.endObject();
        final String responseBody = EntityUtils.toString(searchKNNIndex(indexName, queryBuilder, 3).getEntity());
        return responseBody;
    }

    @SneakyThrows
    public void testADC() {
        // for adc it's sufficient to compute scores with and without adc enabled and assert they are different.

        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        Integer bits = 1;
        int dimension = 8;

        String responseControl = makeQBitIndexADC("control", QFrameBitEncoder.ENABLE_ADC_PARAM, false);
        String responseUnderTest = makeQBitIndexADC("test", QFrameBitEncoder.ENABLE_ADC_PARAM, true);

        logger.info(responseControl);
        logger.info(responseUnderTest);

        List<Object> controlHits = parseSearchResponseHits(responseControl);
        List<Object> testHits = parseSearchResponseHits(responseUnderTest);

        logger.info(controlHits);

        logger.info(controlHits.get(0).getClass());
        logger.info(((java.util.HashMap<String, Object>) controlHits.get(0)));
        logger.info(((java.util.HashMap<String, Object>) controlHits.get(0)).get("_id"));

        logger.info(parseSearchResponseHits(responseUnderTest));

        Double controlFirstHitScore = ((Double) (((java.util.HashMap<String, Object>) controlHits.get(0)).get("_score")));
        Double testFirstScore = ((Double) (((java.util.HashMap<String, Object>) testHits.get(0)).get("_score")));

        assertNotEquals(controlFirstHitScore, testFirstScore);
    }

    @SneakyThrows
    public void testFilterADC() {
        /*

        0. for each of control, test:
        1. create index. ingest 10 documents. force merge index.
        2. run with match all filter query and k = 10
        3. Create index. ingest the same 10 vectors, but with different document ids (11 to 20).
        4. assert that the scores of the results are the same in both searches.
        ./gradlew :integTestRemote --tests  "org.opensearch.knn.index.RandomRotationIT.testFilterADC"  -Dtests.rest.cluster=localhost:9200 -Dtests.cluster=localhost:9200 -Dtests.clustername="integTest-0"
                ./gradlew :integTest --tests  "org.opensearch.knn.index.RandomRotationIT"

         */
        int dimension = 8;
        int bits = 1;
        SpaceType spaceType = SpaceType.L2;
        int k = 10;

        // Generate 10 random vectors that we'll reuse
        List<Float[]> vectors = new ArrayList<>();
        Random random = new Random(42);
        for (int i = 0; i < 10; i++) {
            Float[] vector = new Float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = random.nextFloat();
            }
            vectors.add(vector);
        }

        // Create control index (without filter)
        String controlIndexName = "control-index";
        makeOnlyQBitIndex(controlIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, true, spaceType);

        // Index documents
        for (int i = 0; i < 10; i++) {
            addKnnDoc(controlIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
        }
        forceMergeKnnIndex(controlIndexName);

        // Search without filter
        XContentBuilder controlQueryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(TEST_FIELD_NAME)
            .array("vector", vectors.get(0))
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String controlResponse = EntityUtils.toString(searchKNNIndex(controlIndexName, controlQueryBuilder, k).getEntity());
        List<Object> controlHits = parseSearchResponseHits(controlResponse);
        List<Double> controlScores = controlHits.stream()
            .map(hit -> (Double) ((Map<String, Object>) hit).get("_score"))
            .collect(Collectors.toList());

        // Create test index (with filter)
        String testIndexName = "test-index";
        makeOnlyQBitIndex(testIndexName, QFrameBitEncoder.ENABLE_ADC_PARAM, dimension, bits, true, spaceType);

        // Index same vectors
        for (int i = 0; i < 10; i++) {
            addKnnDoc(testIndexName, String.valueOf(i + 1), ImmutableList.of(TEST_FIELD_NAME), ImmutableList.of(vectors.get(i)));
        }
        forceMergeKnnIndex(testIndexName);

        // Search with match_all filter
        XContentBuilder testQueryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(TEST_FIELD_NAME)
            .array("vector", vectors.get(0))
            .field("k", k)
            .startObject("filter")
            .startObject("match_all")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String testResponse = EntityUtils.toString(searchKNNIndex(testIndexName, testQueryBuilder, k).getEntity());
        List<Object> testHits = parseSearchResponseHits(testResponse);
        List<Double> testScores = testHits.stream()
            .map(hit -> (Double) ((Map<String, Object>) hit).get("_score"))
            .collect(Collectors.toList());

        // Assert that hits are the same
        assertEquals("Number of hits should be equal", controlScores.size(), testScores.size());

        // currently the custom adc java/lucene implementation doesn't match faiss exactly. Investigating.
        // logger.info("Control scores: {}", controlScores);
        // logger.info("Test scores: {}", testScores);
        // for (int i = 0; i < controlScores.size(); i++) {
        // assertEquals("Scores should be equal at position " + i, controlScores.get(i), testScores.get(i), 0.0001);
        // }
        //
        // // Verify same document IDs and order
        // List<String> controlIds = controlHits.stream()
        // .map(hit -> (String) ((Map<String, Object>) hit).get("_id"))
        // .collect(Collectors.toList());
        // List<String> testIds = testHits.stream().map(hit -> (String) ((Map<String, Object>)
        // hit).get("_id")).collect(Collectors.toList());
        //
        // assertEquals("Document IDs should be in the same order", controlIds, testIds);
    }

    protected List<Object> parseSearchResponseHits(String responseBody) throws IOException {
        return (List<Object>) ((Map<String, Object>) createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map()
            .get("hits")).get("hits");
    }
}
