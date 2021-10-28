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
import org.apache.http.util.EntityUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.action.ActionFuture;
import org.opensearch.action.ActionListener;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.GetModelAction;
import org.opensearch.knn.plugin.transport.GetModelRequest;
import org.opensearch.knn.plugin.transport.GetModelResponse;
import org.opensearch.rest.RestStatus;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ERROR;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.common.KNNConstants.MODEL_TIMESTAMP;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * Integration tests to check the correctness of {@link org.opensearch.knn.plugin.rest.RestGetModelHandler}
 */

public class RestGetModelHandlerIT extends KNNRestTestCase {

    private ModelMetadata getModelMetadata() {
        return new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, 4, ModelState.CREATED,
            "2021-03-27", "test model", "");
    }

    public void testGetModelExists() throws IOException {
        createModelSystemIndex();
        String testModelID = "test-model-id";
        byte[] testModelBlob = "hello".getBytes();
        ModelMetadata testModelMetadata = getModelMetadata();

        addModelToSystemIndex(testModelID, testModelMetadata, testModelBlob);

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, testModelID);
        Request request = new Request("GET", restURI);

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(
            XContentType.JSON.xContent(),
            responseBody
        ).map();

        assertEquals(testModelID, responseMap.get(MODEL_ID));
        assertEquals(testModelMetadata.getDescription(), responseMap.get(MODEL_DESCRIPTION));
        assertEquals(testModelMetadata.getDimension(), responseMap.get(DIMENSION));
        assertEquals(testModelMetadata.getError(), responseMap.get(MODEL_ERROR));
        assertEquals(testModelMetadata.getKnnEngine().getName(), responseMap.get(KNN_ENGINE));
        assertEquals(testModelMetadata.getSpaceType().getValue(), responseMap.get(METHOD_PARAMETER_SPACE_TYPE));
        assertEquals(testModelMetadata.getState().getName(), responseMap.get(MODEL_STATE));
        assertEquals(testModelMetadata.getTimestamp(), responseMap.get(MODEL_TIMESTAMP));
    }



    public void testGetModelExistsWithFilter() throws IOException {
        createModelSystemIndex();
        String testModelID = "test-model-id";
        byte[] testModelBlob = "hello".getBytes();
        ModelMetadata testModelMetadata = getModelMetadata();

        addModelToSystemIndex(testModelID, testModelMetadata, testModelBlob);

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, testModelID);
        Request request = new Request("GET", restURI);

        List<String> filterdPath = Arrays.asList(MODEL_ID,MODEL_DESCRIPTION,MODEL_TIMESTAMP,KNN_ENGINE);
        request.addParameter("filter_path", Strings.join(filterdPath, ","));

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(
            XContentType.JSON.xContent(),
            responseBody
        ).map();

        assertTrue(responseMap.size()==filterdPath.size());
        assertEquals(testModelID, responseMap.get(MODEL_ID));
        assertEquals(testModelMetadata.getDescription(), responseMap.get(MODEL_DESCRIPTION));
        assertEquals(testModelMetadata.getTimestamp(), responseMap.get(MODEL_TIMESTAMP));
        assertEquals(testModelMetadata.getKnnEngine().getName(), responseMap.get(KNN_ENGINE));
        assertFalse(responseMap.containsKey(DIMENSION));
        assertFalse(responseMap.containsKey(MODEL_ERROR));
        assertFalse(responseMap.containsKey(METHOD_PARAMETER_SPACE_TYPE));
        assertFalse(responseMap.containsKey(MODEL_STATE));
    }

    public void testGetModelFailsInvalid() throws IOException {
        createModelSystemIndex();
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, "invalid-model-id");
        Request request = new Request("GET", restURI);

        ResponseException ex = expectThrows(ResponseException.class, () ->
            client().performRequest(request));
        assertTrue(ex.getMessage().contains("\"invalid-model-id\""));
    }

    public void testGetModelFailsBlank() throws IOException {
        createModelSystemIndex();
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, " ");
        Request request = new Request("GET", restURI);

        expectThrows(IllegalArgumentException.class, () -> client().performRequest(request));
    }
}
