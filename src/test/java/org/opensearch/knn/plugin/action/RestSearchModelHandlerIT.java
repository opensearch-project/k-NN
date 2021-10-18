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

import org.apache.http.util.EntityUtils;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.rest.RestStatus;
import org.opensearch.search.SearchHit;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.MODELS;


/**
 * Integration tests to check the correctness of {@link org.opensearch.knn.plugin.rest.RestSearchModelHandler}
 */

public class RestSearchModelHandlerIT extends KNNRestTestCase {

    private ModelMetadata getModelMetadata() {
        return new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, 4, ModelState.CREATED,
            "2021-03-27", "test model", "");
    }

    public void testNotSupportedParams() throws IOException {
        createModelSystemIndex();
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, "_search");
        Map<String,String> invalidParams = new HashMap<>();
        invalidParams.put("index", "index-name");
        Request request = new Request("GET", restURI);
        request.addParameters(invalidParams);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
    }


    public void testNoModelExists() throws IOException {
        createModelSystemIndex();
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, "_search");
        Request request = new Request("GET", restURI);
        request.setJsonEntity("{\n" +
            "    \"query\": {\n" +
            "        \"match_all\": {}\n" +
            "    }\n" +
            "}");

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        String responseBody = EntityUtils.toString(response.getEntity());
        assertNotNull(responseBody);

        XContentParser parser = createParser(XContentType.JSON.xContent(), responseBody);
        SearchResponse searchResponse = SearchResponse.fromXContent(parser);
        assertNotNull(searchResponse);
        assertEquals(searchResponse.getHits().getHits().length, 0);

    }

    public void testSearchModelExists() throws IOException {
        createModelSystemIndex();
        createIndex("irrelevant-index", Settings.EMPTY);
        addDocWithBinaryField("irrelevant-index", "id1", "field-name", "value");
        List<String> testModelID = Arrays.asList("test-modelid1", "test-modelid2");
        byte[] testModelBlob = "hello".getBytes();
        ModelMetadata testModelMetadata = getModelMetadata();
        for(String modelID: testModelID){
            addModelToSystemIndex(modelID, testModelMetadata, testModelBlob);
        }

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, "_search");

        for(String method: Arrays.asList("GET", "POST")){
            Request request = new Request(method, restURI);
            request.setJsonEntity("{\n" +
                "    \"query\": {\n" +
                "        \"match_all\": {}\n" +
                "    }\n" +
                "}");
            Response response = client().performRequest(request);
            assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

            String responseBody = EntityUtils.toString(response.getEntity());
            assertNotNull(responseBody);

            XContentParser parser = createParser(XContentType.JSON.xContent(), responseBody);
            SearchResponse searchResponse = SearchResponse.fromXContent(parser);
            assertNotNull(searchResponse);

            //returns only model from ModelIndex
            assertEquals(searchResponse.getHits().getHits().length, testModelID.size());

            for(SearchHit hit: searchResponse.getHits().getHits()){
                assertTrue(testModelID.contains(hit.getId()));
            }
        }
    }
}
