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

import joptsimple.internal.Strings;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.core.xcontent.MediaTypeParserRegistry;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.core.rest.RestStatus;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ERROR;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.common.KNNConstants.MODEL_TIMESTAMP;
import static org.opensearch.knn.index.SpaceType.L2;
import static org.opensearch.knn.index.util.KNNEngine.FAISS;

/**
 * Integration tests to check the correctness of {@link org.opensearch.knn.plugin.rest.RestGetModelHandler}
 */

public class RestGetModelHandlerIT extends KNNRestTestCase {

    public void testGetModelExists() throws Exception {
        createModelSystemIndex();

        String modelId = "test-model-id";
        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";
        int dimension = 8;
        String modelDescription = "dummy description";

        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);

        ingestDataAndTrainModel(
            modelId,
            trainingIndexName,
            trainingFieldName,
            dimension,
            modelDescription,
            xContentBuilderToMap(getModelMethodBuilder())
        );
        assertTrainingSucceeds(modelId, NUM_OF_ATTEMPTS, DELAY_MILLI_SEC);

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, modelId);
        Request request = new Request("GET", restURI);

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeParserRegistry.getDefaultMediaType().xContent(), responseBody).map();
        assertEquals(modelId, responseMap.get(MODEL_ID));
        assertEquals(modelDescription, responseMap.get(MODEL_DESCRIPTION));
        assertEquals(FAISS.getName(), responseMap.get(KNN_ENGINE));
        assertEquals(L2.getValue(), responseMap.get(METHOD_PARAMETER_SPACE_TYPE));
    }

    public void testGetModelExistsWithFilter() throws Exception {
        createModelSystemIndex();
        String modelId = "test-model-id";
        String trainingIndexName = "train-index";
        String trainingFieldName = "train-field";
        int dimension = 8;
        String modelDescription = "dummy description";

        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        Map<String, Object> method = xContentBuilderToMap(getModelMethodBuilder());
        ingestDataAndTrainModel(modelId, trainingIndexName, trainingFieldName, dimension, modelDescription, method);
        assertTrainingSucceeds(modelId, NUM_OF_ATTEMPTS, DELAY_MILLI_SEC);

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, modelId);
        Request request = new Request("GET", restURI);

        List<String> filteredPath = Arrays.asList(MODEL_ID, MODEL_DESCRIPTION, MODEL_TIMESTAMP, KNN_ENGINE);
        request.addParameter("filter_path", Strings.join(filteredPath, ","));

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeParserRegistry.getDefaultMediaType().xContent(), responseBody).map();

        assertTrue(responseMap.size() == filteredPath.size());
        assertEquals(modelId, responseMap.get(MODEL_ID));
        assertEquals(modelDescription, responseMap.get(MODEL_DESCRIPTION));
        assertEquals(FAISS.getName(), responseMap.get(KNN_ENGINE));
        assertFalse(responseMap.containsKey(DIMENSION));
        assertFalse(responseMap.containsKey(MODEL_ERROR));
        assertFalse(responseMap.containsKey(METHOD_PARAMETER_SPACE_TYPE));
        assertFalse(responseMap.containsKey(MODEL_STATE));
    }

    public void testGetModelFailsInvalid() throws IOException {
        createModelSystemIndex();
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, "invalid-model-id");
        Request request = new Request("GET", restURI);

        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(ex.getMessage().contains("\"invalid-model-id\""));
    }

    public void testGetModelFailsBlank() throws IOException {
        createModelSystemIndex();
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, " ");
        Request request = new Request("GET", restURI);

        expectThrows(IllegalArgumentException.class, () -> client().performRequest(request));
    }
}
