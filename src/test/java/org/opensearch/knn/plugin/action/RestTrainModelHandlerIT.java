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
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.core.rest.RestStatus;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;

public class RestTrainModelHandlerIT extends KNNRestTestCase {

    public void testTrainModel_fail_notEnoughData() throws Exception {

        // Check that training fails properly when there is not enough data

        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";
        int dimension = 16;

        // Create a training index and randomly ingest data into it
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        int trainingDataCount = 4;
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        // Call the train API with this definition:
        /*
            {
              "training_index": "train_index",
              "training_field": "train_field",
              "dimension": 16,
              "description": "this should be allowed to be null",
              "method": {
                  "name":"ivf",
                  "engine":"faiss",
                  "space_type": "innerproduct",
                  "parameters":{
                    "nlist":128,
                    "encoder":{
                        "name":"pq",
                        "parameters":{
                            "code_size":2,
                            "code_count": 2
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
            .field(METHOD_PARAMETER_SPACE_TYPE, "innerproduct")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 128)
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

        Response trainResponse = trainModel(null, trainingIndexName, trainingFieldName, dimension, method, "dummy description");

        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));

        // Grab the model id from the response
        String trainResponseBody = EntityUtils.toString(trainResponse.getEntity());
        assertNotNull(trainResponseBody);

        Map<String, Object> trainResponseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), trainResponseBody).map();
        String modelId = (String) trainResponseMap.get(MODEL_ID);
        assertNotNull(modelId);

        // Confirm that the model fails to create
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        assertEquals(modelId, responseMap.get(MODEL_ID));

        assertTrainingFails(modelId, 30, 1000);
    }

    public void testTrainModel_fail_tooMuchData() throws Exception {
        // Limit the cache size and then call train

        updateClusterSettings("knn.memory.circuit_breaker.limit", "1kb");

        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";
        int dimension = 16;

        // Create a training index and randomly ingest data into it
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        int trainingDataCount = 20; // 20 * 16 * 4 ~= 10 kb
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        // Call the train API with this definition:
        /*
            {
              "training_index": "train_index",
              "training_field": "train_field",
              "dimension": 16,
              "description": "this should be allowed to be null",
              "method": {
                  "name":"ivf",
                  "engine":"faiss",
                  "space_type": "innerproduct",
                  "parameters":{
                    "nlist":128,
                    "encoder":{
                        "name":"pq",
                        "parameters":{
                            "code_size":2,
                            "code_count": 2
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
            .field(METHOD_PARAMETER_SPACE_TYPE, "innerproduct")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 128)
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

        Response trainResponse = trainModel(null, trainingIndexName, trainingFieldName, dimension, method, "dummy description");

        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));

        // Grab the model id from the response
        String trainResponseBody = EntityUtils.toString(trainResponse.getEntity());
        assertNotNull(trainResponseBody);

        Map<String, Object> trainResponseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), trainResponseBody).map();
        String modelId = (String) trainResponseMap.get(MODEL_ID);
        assertNotNull(modelId);

        // Confirm that the model fails to create
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        assertEquals(modelId, responseMap.get(MODEL_ID));

        assertTrainingFails(modelId, 30, 1000);
    }

    public void testTrainModel_fail_commaInDescription() throws Exception {
        // Test checks that training when passing in an id succeeds

        String modelId = "test-model-id";
        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";
        int dimension = 8;

        // Create a training index and randomly ingest data into it
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);

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

        Exception e = expectThrows(
            ResponseException.class,
            () -> trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, "dummy description, with comma")
        );
        assertTrue(e.getMessage().contains("Model description cannot contain any commas: ','"));
    }

    public void testTrainModel_success_withId() throws Exception {
        // Test checks that training when passing in an id succeeds

        String modelId = "test-model-id";
        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";
        int dimension = 8;

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

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        assertEquals(modelId, responseMap.get(MODEL_ID));

        // Make sure training succeeds after 30 seconds
        assertTrainingSucceeds(modelId, 30, 1000);
    }

    public void testTrainModel_success_noId() throws Exception {
        // Test to check if training succeeds when no id is passed in

        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";
        int dimension = 16;

        // Create a training index and randomly ingest data into it
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        int trainingDataCount = 150;
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        // Call the train API with this definition:
        /*
            {
              "training_index": "train_index",
              "training_field": "train_field",
              "dimension": 16,
              "description": "this should be allowed to be null",
              "method": {
                  "name":"ivf",
                  "engine":"faiss",
                  "space_type": "innerproduct",
                  "parameters":{
                    "nlist":2,
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
            .field(METHOD_PARAMETER_SPACE_TYPE, "innerproduct")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 2)
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

        Response trainResponse = trainModel(null, trainingIndexName, trainingFieldName, dimension, method, "dummy description");

        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));

        // Grab the model id from the response
        String trainResponseBody = EntityUtils.toString(trainResponse.getEntity());
        assertNotNull(trainResponseBody);

        Map<String, Object> trainResponseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), trainResponseBody).map();
        String modelId = (String) trainResponseMap.get(MODEL_ID);
        assertNotNull(modelId);

        // Confirm that the model gets created
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        assertEquals(modelId, responseMap.get(MODEL_ID));

        assertTrainingSucceeds(modelId, 30, 1000);
    }

    // Test to checks when user tries to train a model with nested fields
    public void testTrainModel_success_nestedField() throws Exception {
        String modelId = "test-model-id";
        String trainingIndexName = "train-index";
        String nestedFieldPath = "a.b.train-field";
        int dimension = 8;

        // Create a training index and randomly ingest data into it
        String mapping = createKnnIndexNestedMapping(dimension, nestedFieldPath);
        createKnnIndex(trainingIndexName, mapping);
        int trainingDataCount = 200;
        bulkIngestRandomVectorsWithNestedField(trainingIndexName, nestedFieldPath, trainingDataCount, dimension);

        // Call the train API with this definition:
        /*
            {
                "training_index": "train_index",
                "training_field": "a.b.train_field",
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

        Response trainResponse = trainModel(modelId, trainingIndexName, nestedFieldPath, dimension, method, "dummy description");

        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));

        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        assertEquals(modelId, responseMap.get(MODEL_ID));

        assertTrainingSucceeds(modelId, 30, 1000);
    }

    // Test to checks when user tries to train a model compression/mode and method
    public void testTrainModel_success_methodOverrideWithCompressionMode() throws Exception {
        String modelId = "test-model-id";
        String trainingIndexName = "train-index";
        String nestedFieldPath = "a.b.train-field";
        int dimension = 8;

        // Create a training index and randomly ingest data into it
        String mapping = createKnnIndexNestedMapping(dimension, nestedFieldPath);
        createKnnIndex(trainingIndexName, mapping);
        int trainingDataCount = 200;
        bulkIngestRandomVectorsWithNestedField(trainingIndexName, nestedFieldPath, trainingDataCount, dimension);

        // Call the train API with this definition:

        /*
        POST /_plugins/_knn/models/test-model/_train
            {
                "training_index": "train_index",
                "training_field": "train_field",
                "dimension": 8,
                "description": "model",
                "space_type": "innerproduct",
                "mode": "on_disk",
                "method": {
                    "name": "ivf",
                    "params": {
                      "nlist": 16
                    }
                  }
            }

         */
        XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .field(NAME, "ivf")
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_NLIST, 16)
                .endObject()
                .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        XContentBuilder outerParams = XContentFactory.jsonBuilder()
                .startObject()
                .field(TRAIN_INDEX_PARAMETER, trainingIndexName)
                .field(TRAIN_FIELD_PARAMETER, nestedFieldPath)
                .field(DIMENSION, dimension)
                .field(COMPRESSION_LEVEL_PARAMETER, "16x")
                .field(MODE_PARAMETER, "on_disk")
                .field(KNN_METHOD, method)
                .field(MODEL_DESCRIPTION, "dummy description")
                .endObject();

        Request request = new Request("POST", "/_plugins/_knn/models/" + modelId + "/_train");
        request.setJsonEntity(outerParams.toString());

        Response trainResponse = client().performRequest(request);

        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));

        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();

        assertEquals(modelId, responseMap.get(MODEL_ID));

        assertTrainingSucceeds(modelId, 30, 1000);
    }
}
