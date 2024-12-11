/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.AfterClass;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.search.SearchHit;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import static org.opensearch.knn.TestUtils.KNN_BWC_PREFIX;
import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;

public class ModelIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_MODEL_INDEX = KNN_BWC_PREFIX + "test-model-index";
    private static final String TEST_MODEL_INDEX_DEFAULT = KNN_BWC_PREFIX + "test-model-index-default";
    private static final String TRAINING_INDEX = KNN_BWC_PREFIX + "train-index";
    private static final String TRAINING_INDEX_DEFAULT = KNN_BWC_PREFIX + "train-index-default";
    private static final String TRAINING_INDEX_FOR_NON_KNN_INDEX = KNN_BWC_PREFIX + "train-index-for-non-knn-index";
    private static final String TRAINING_FIELD = "train-field";
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static int DOC_ID_TEST_MODEL_INDEX = 0;
    private static int DOC_ID_TEST_MODEL_INDEX_DEFAULT = 0;
    private static final int DELAY_MILLI_SEC = 1000;
    private static final int MIN_NUM_OF_MODELS = 2;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static final int NUM_DOCS_TEST_MODEL_INDEX = 100;
    private static final int NUM_DOCS_TEST_MODEL_INDEX_DEFAULT = 100;
    private static final int NUM_DOCS_TEST_MODEL_INDEX_FOR_NON_KNN_INDEX = 100;
    private static final int NUM_OF_ATTEMPTS = 30;
    private static int QUERY_COUNT = 0;
    private static int QUERY_COUNT_TEST_MODEL_INDEX = 0;
    private static int QUERY_COUNT_TEST_MODEL_INDEX_DEFAULT = 0;
    private static final String TEST_MODEL_ID = "test-model-id";
    private static final String TEST_MODEL_ID_DEFAULT = "test-model-id-default";
    private static final String TEST_MODEL_ID_FOR_NON_KNN_INDEX = "test-model-id-for-non-knn-index";
    private static final String MODEL_DESCRIPTION = "Description for train model test";

    // KNN model test
    public void testKNNModel() throws Exception {
        if (isRunningAgainstOldCluster()) {

            // Create a training index and randomly ingest data into it
            createBasicKnnIndex(TRAINING_INDEX, TRAINING_FIELD, DIMENSIONS);
            bulkIngestRandomVectors(TRAINING_INDEX, TRAINING_FIELD, NUM_DOCS, DIMENSIONS);

            trainKNNModel(TEST_MODEL_ID, TRAINING_INDEX, TRAINING_FIELD, DIMENSIONS, MODEL_DESCRIPTION);
            validateModelCreated(TEST_MODEL_ID);

            createKnnIndex(testIndex, modelIndexMapping(TEST_FIELD, TEST_MODEL_ID));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            Thread.sleep(1000);
            DOC_ID = NUM_DOCS;
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = 2 * NUM_DOCS;
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);

            searchKNNModel(TEST_MODEL_ID);

            createKnnIndex(TEST_MODEL_INDEX, modelIndexMapping(TEST_FIELD, TEST_MODEL_ID));
            addKNNDocs(TEST_MODEL_INDEX, TEST_FIELD, DIMENSIONS, DOC_ID_TEST_MODEL_INDEX, NUM_DOCS_TEST_MODEL_INDEX);
            QUERY_COUNT_TEST_MODEL_INDEX = NUM_DOCS_TEST_MODEL_INDEX;
            validateKNNSearch(TEST_MODEL_INDEX, TEST_FIELD, DIMENSIONS, QUERY_COUNT_TEST_MODEL_INDEX, K);

            deleteKNNIndex(testIndex);
            deleteKNNIndex(TRAINING_INDEX);
            deleteKNNIndex(TEST_MODEL_INDEX);
        }
    }

    // KNN model test Default Parameters
    public void testKNNModelDefault() throws Exception {
        if (isRunningAgainstOldCluster()) {

            // Create a training index and randomly ingest data into it
            createBasicKnnIndex(TRAINING_INDEX_DEFAULT, TRAINING_FIELD, DIMENSIONS);
            bulkIngestRandomVectors(TRAINING_INDEX_DEFAULT, TRAINING_FIELD, NUM_DOCS, DIMENSIONS);

            trainKNNModel(TEST_MODEL_ID_DEFAULT, TRAINING_INDEX_DEFAULT, TRAINING_FIELD, DIMENSIONS, MODEL_DESCRIPTION);
            validateModelCreated(TEST_MODEL_ID_DEFAULT);

            createKnnIndex(testIndex, modelIndexMapping(TEST_FIELD, TEST_MODEL_ID_DEFAULT));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            Thread.sleep(1000);
            DOC_ID = NUM_DOCS;
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = 2 * NUM_DOCS;
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);

            searchKNNModel(TEST_MODEL_ID);

            createKnnIndex(TEST_MODEL_INDEX_DEFAULT, modelIndexMapping(TEST_FIELD, TEST_MODEL_ID_DEFAULT));
            addKNNDocs(
                TEST_MODEL_INDEX_DEFAULT,
                TEST_FIELD,
                DIMENSIONS,
                DOC_ID_TEST_MODEL_INDEX_DEFAULT,
                NUM_DOCS_TEST_MODEL_INDEX_DEFAULT
            );
            QUERY_COUNT_TEST_MODEL_INDEX_DEFAULT = NUM_DOCS_TEST_MODEL_INDEX_DEFAULT;
            validateKNNSearch(TEST_MODEL_INDEX_DEFAULT, TEST_FIELD, DIMENSIONS, QUERY_COUNT_TEST_MODEL_INDEX_DEFAULT, K);

            deleteKNNIndex(testIndex);
            deleteKNNIndex(TRAINING_INDEX_DEFAULT);
            deleteKNNIndex(TEST_MODEL_INDEX_DEFAULT);
        }
    }

    public void testNonKNNIndex_withModelId() throws Exception {
        if (isRunningAgainstOldCluster()) {

            // Create a training index and randomly ingest data into it
            createBasicKnnIndex(TRAINING_INDEX_FOR_NON_KNN_INDEX, TRAINING_FIELD, DIMENSIONS);
            bulkIngestRandomVectors(TRAINING_INDEX_FOR_NON_KNN_INDEX, TRAINING_FIELD, NUM_DOCS, DIMENSIONS);

            trainKNNModel(TEST_MODEL_ID_FOR_NON_KNN_INDEX, TRAINING_INDEX_FOR_NON_KNN_INDEX, TRAINING_FIELD, DIMENSIONS, MODEL_DESCRIPTION);
            validateModelCreated(TEST_MODEL_ID_FOR_NON_KNN_INDEX);

            createKnnIndex(
                testIndex,
                createKNNDefaultScriptScoreSettings(),
                modelIndexMapping(TEST_FIELD, TEST_MODEL_ID_FOR_NON_KNN_INDEX)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            Thread.sleep(1000);
            DOC_ID = NUM_DOCS;
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            deleteKNNIndex(testIndex);
            deleteKNNIndex(TRAINING_INDEX_FOR_NON_KNN_INDEX);
            deleteKNNModel(TEST_MODEL_ID_FOR_NON_KNN_INDEX);
        }
    }

    // Delete Models and ".opensearch-knn-models" index to clear cluster metadata
    @AfterClass
    public static void wipeAllModels() throws IOException {
        if (!isRunningAgainstOldCluster()) {
            deleteKNNModel(TEST_MODEL_ID);
            deleteKNNModel(TEST_MODEL_ID_DEFAULT);
        }
    }

    // Delete model by taking modelId as input parameter
    public static void deleteKNNModel(String modelId) throws IOException {
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, modelId);
        Request request = new Request("DELETE", restURI);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    public void searchKNNModel(String testModelID) throws Exception {
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, "_search");

        for (String method : Arrays.asList("GET", "POST")) {
            Request request = new Request(method, restURI);
            request.setJsonEntity("{\n" + "\"_source\" : false,\n" + "\"query\": {\n" + "\"match_all\": {}\n" + "}\n" + "}");
            Response response = client().performRequest(request);
            assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

            String responseBody = EntityUtils.toString(response.getEntity());
            assertNotNull(responseBody);

            XContentParser parser = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody);
            SearchResponse searchResponse = SearchResponse.fromXContent(parser);
            assertNotNull(searchResponse);
            assertTrue(MIN_NUM_OF_MODELS <= searchResponse.getHits().getHits().length);

            for (SearchHit hit : searchResponse.getHits().getHits()) {
                assertTrue(hit.getId().startsWith(testModelID));
            }
        }
    }

    // Confirm that the model gets created using Get Model API
    public void validateModelCreated(String modelId) throws Exception {
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        assertEquals(modelId, responseMap.get(MODEL_ID));
        assertTrainingSucceeds(modelId, NUM_OF_ATTEMPTS, DELAY_MILLI_SEC);
    }

    // train KNN model
    // method : "ivf", engine : "faiss", space_type : "l2", nlists : 1
    public void trainKNNModel(String modelId, String trainingIndexName, String trainingFieldName, int dimension, String description)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        Response trainResponse = trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, description);
        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
    }

    // train KNN model Default Parameters
    // method : "ivf", engine : "nmslib", space_type : "l2"
    public void trainKNNModelDefault(String modelId, String trainingIndexName, String trainingFieldName, int dimension, String description)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, NMSLIB_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        Response trainResponse = trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, description);
        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
    }

    // mapping to create index from model
    public String modelIndexMapping(String fieldName, String modelId) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(fieldName)
            .field(VECTOR_TYPE, KNN_VECTOR)
            .field(MODEL_ID, modelId)
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }
}
