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

package org.opensearch.knn.plugin.action;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.DeleteModelResponse;
import org.opensearch.rest.RestStatus;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * Integration tests to check the correctness of {@link org.opensearch.knn.plugin.rest.RestDeleteModelHandler}
 */

public class RestDeleteModelHandlerIT extends KNNRestTestCase {

    private ModelMetadata getModelMetadata() {
        return new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, 4, ModelState.CREATED, "2021-03-27", "test model", "");
    }

    public void testDeleteModelExists() throws Exception {
        createModelSystemIndex();
        String testModelID = "test-model-id";
        byte[] testModelBlob = "hello".getBytes();
        ModelMetadata testModelMetadata = getModelMetadata();

        addModelToSystemIndex(testModelID, testModelMetadata, testModelBlob);
        assertEquals(getDocCount(MODEL_INDEX_NAME), 1);

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, testModelID);
        Request request = new Request("DELETE", restURI);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        assertEquals(0, getDocCount(MODEL_INDEX_NAME));
    }

    public void testDeleteTrainingModel() throws Exception {
        createModelSystemIndex();
        String testModelID = "test-model-id";
        byte[] testModelBlob = "hello".getBytes();
        ModelMetadata testModelMetadata = getModelMetadata();
        testModelMetadata.setState(ModelState.TRAINING);

        addModelToSystemIndex(testModelID, testModelMetadata, testModelBlob);
        assertEquals(1, getDocCount(MODEL_INDEX_NAME));

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, testModelID);
        Request request = new Request("DELETE", restURI);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        assertEquals(1, getDocCount(MODEL_INDEX_NAME));

        String responseBody = EntityUtils.toString(response.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();

        assertEquals(testModelID, responseMap.get(MODEL_ID));
        assertEquals("failed", responseMap.get(DeleteModelResponse.RESULT));

        String errorMessage = String.format("Cannot delete model \"%s\". Model is still in training", testModelID);
        assertEquals(errorMessage, responseMap.get(DeleteModelResponse.ERROR_MSG));
    }

    public void testDeleteModelFailsInvalid() throws Exception {
        String modelId = "invalid-model-id";
        createModelSystemIndex();
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, modelId);
        Request request = new Request("DELETE", restURI);

        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(ex.getMessage().contains(modelId));
    }

    // Test Train Model -> Delete Model -> Train Model with same modelId
    public void testTrainingDeletedModel() throws Exception {
        String modelId = "test-model-id1";
        String trainingIndexName1 = "train-index-1";
        String trainingIndexName2 = "train-index-2";
        String trainingFieldName = "train-field";
        int dimension = 8;

        // Train Model
        trainModel(modelId, trainingIndexName1, trainingFieldName, dimension);

        // Delete Model
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, modelId);
        Request request = new Request("DELETE", restURI);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        assertEquals(0, getDocCount(MODEL_INDEX_NAME));

        // Train Model again with same ModelId
        trainModel(modelId, trainingIndexName2, trainingFieldName, dimension);
    }

    private void trainModel(String modelId, String trainingIndexName, String trainingFieldName, int dimension) throws Exception {

        // Create a training index and randomly ingest data into it
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        int trainingDataCount = 200;
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        // Call the train API with this definition:
        /*
            {
              "training_index": "train_index",
              "training_field": "train_field",
              "dimension": 8,
              "description": "this should be allowed to be null",
              "method": {
                  "name":"ivf",
                  "engine":"faiss",
                  "space_type": "l2",
                  "parameters":{
                    "nlist":1,
                    "encoder":{
                        "name":"pq",
                        "parameters":{
                            "code_size":2,
                            "m": 2
                        }
                    }
                  }
              }
            }
        */
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, "ivf")
            .field(KNN_ENGINE, "faiss")
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, "pq")
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_CODE_SIZE, 2)
            .field(ENCODER_PARAMETER_PQ_M, 2)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        Response trainResponse = trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, "dummy description");

        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));

        // Confirm that the model gets created
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();

        assertEquals(modelId, responseMap.get(MODEL_ID));

        // Make sure training succeeds after 30 seconds
        assertTrainingSucceeds(modelId, 30, 1000);
    }

}
