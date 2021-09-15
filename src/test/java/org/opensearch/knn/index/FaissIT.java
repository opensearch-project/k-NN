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
// import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Floats;
import org.apache.http.util.EntityUtils;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.util.KNNEngine;
// import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

// import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
// import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
// import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
// import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class FaissIT extends KNNRestTestCase {

    static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        URL testIndexVectors = FaissIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = FaissIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath());
    }

    public void testEndToEnd_fromMethod() throws IOException, InterruptedException {
        String indexName = "test-index-1";
        String fieldName = "test-field-1";

        KNNMethod hnswMethod = KNNEngine.FAISS.getMethod(KNNConstants.METHOD_HNSW);
        SpaceType spaceType = SpaceType.INNER_PRODUCT;

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
        String mapping = Strings.toString(builder);

        createKnnIndex(indexName, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(indexName)));

        // Index the test data
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(indexName, Integer.toString(testData.indexData.docs[i]), fieldName,
                    Floats.asList(testData.indexData.vectors[i]).toArray());
        }

        // Assert we have the right number of documents in the index
        refreshAllIndices();
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
                assertEquals(KNNEngine.FAISS.score(KNNScoringUtil.innerProduct(testData.queries[i], primitiveArray),
                        spaceType), actualScores.get(j), 0.0001);
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

            Thread.sleep(5*1000);
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

        String mapping = Strings.toString(builder);
        createKnnIndex(indexName, mapping);

        Float[] vector  = {6.0f, 6.0f};
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // update
        Float[] updatedVector  = {8.0f, 8.0f};
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

        String mapping = Strings.toString(builder);
        createKnnIndex(indexName, mapping);

        Float[] vector  = {6.0f, 6.0f};
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, vector);

        // delete knn doc
        deleteKnnDoc(INDEX_NAME, "1");
    }

    public void testEndToEnd_fromModel() throws IOException {
        //TODO this test is broken. Unfortunately, we can not add a document and add cluster metadata to the index
        // about the document. Once training functionality is added, we will need to use the train api to add the
        // model to the cluster
//        String modelId = "test-model";
//        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128);
//
//        // Create model -- just runs a brute force search
//        Map<String, Object> params = ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, FAISS_FLAT_DESCRIPTION,
//                SPACE_TYPE, modelMetadata.getSpaceType().getValue());
//        byte[] model = JNIService.trainIndex(params, modelMetadata.getDimension(), 0,
//                modelMetadata.getKnnEngine().getName());
//
//        // Create model system index
//        createModelSystemIndex();
//
//        // Add model to model system index
//        addModelToSystemIndex(modelId, modelMetadata, model);
//
//        // Create knn index from model
//        String fieldName = "test-field-name";
//        String indexName = "test-index-name";
//        String indexMapping = Strings.toString(XContentFactory.jsonBuilder().startObject()
//                .startObject("properties")
//                .startObject(fieldName)
//                .field("type", "knn_vector")
//                .field(MODEL_ID, modelId)
//                .endObject()
//                .endObject()
//                .endObject());
//
//        createKnnIndex(indexName, getKNNDefaultIndexSettings(), indexMapping);
//
//        // Index some documents
//        int numDocs = 100;
//        for (int i = 0; i < numDocs; i++) {
//            Float[] indexVector = new Float[modelMetadata.getDimension()];
//            Arrays.fill(indexVector, (float) i);
//
//            addKnnDoc(indexName, Integer.toString(i), fieldName, indexVector);
//        }
//
//        // Run search and ensure that the values returned are expected
//        float[] queryVector = new float[modelMetadata.getDimension()];
//        Arrays.fill(queryVector, (float) numDocs);
//        int k = 10;
//
//        Response searchResponse = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, queryVector, k), k);
//        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
//
//        for (int i = 0; i < k; i++) {
//            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
//        }
    }
}
