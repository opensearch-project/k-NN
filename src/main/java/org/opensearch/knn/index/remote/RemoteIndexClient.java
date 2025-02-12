/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.NotImplementedException;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.classic.methods.HttpUriRequestBase;
import org.apache.hc.client5.http.impl.classic.BasicHttpClientResponseHandler;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.client5.http.utils.Base64;
import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.opensearch.core.common.settings.SecureString;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;

/**
 * Class to handle all interactions with the remote vector build service.
 * InterruptedExceptions will cause a fallback to local CPU build.
 */
@Log4j2
public class RemoteIndexClient {
    private static RemoteIndexClient INSTANCE;
    private volatile CloseableHttpClient httpClient;
    public static final int MAX_RETRIES = 1; // 2 total attempts
    public static final long BASE_DELAY_MS = 1000;

    private static final ObjectMapper objectMapper = new ObjectMapper();

    RemoteIndexClient() {
        this.httpClient = createHttpClient();
    }

    /**
     * Return the Singleton instance of the node's RemoteIndexClient
     * @return RemoteIndexClient instance
     */
    public static synchronized RemoteIndexClient getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new RemoteIndexClient();
        }
        return INSTANCE;
    }

    /**
     * Initialize the httpClient to be used
     * @return The HTTP Client
     */
    private CloseableHttpClient createHttpClient() {
        // TODO The client will need to be rebuilt iff we decide to allow for retry configuration in the future
        return HttpClients.custom().setRetryStrategy(new RemoteIndexClientRetryStrategy()).build();
    }

    /**
    * Submit a build to the Remote Vector Build Service endpoint using round robin task assignment.
    * @return job_id from the server response used to track the job
    */
    public String submitVectorBuild(RemoteBuildRequest request) throws IOException {
        URI endpoint = URI.create(KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT));
        HttpPost buildRequest = new HttpPost(endpoint + "/_build");
        buildRequest.setHeader("Content-Type", "application/json");
        buildRequest.setEntity(new StringEntity(request.toJson()));
        authenticateRequest(buildRequest);

        String response = httpClient.execute(buildRequest, body -> {
            if (body.getCode() != 200) {
                throw new IOException("Failed to submit build request after retries with code: " + body.getCode());
            }
            return EntityUtils.toString(body.getEntity());
        });

        if (response == null) {
            throw new IOException("Received 200 status code but response is null.");
        }

        return getValueFromResponse(response, "job_id");
    }

    /**
    * Await the completion of the index build by polling periodically and handling the returned statuses.
    * @param jobId identifier from the server to track the job
    * @return the path to the completed index
    */
    private void awaitVectorBuild(String jobId) {
        throw new NotImplementedException();
    }

    /**
     * Helper method to directly get the status response for a given job ID
     * @param jobId to check
     * @return The entire response for the status request
     */
    public String getBuildStatus(String jobId) throws IOException {
        URI endpoint = URI.create(KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT));
        HttpGet request = new HttpGet(endpoint + "/_status/" + jobId);
        authenticateRequest(request);
        return httpClient.execute(request, new BasicHttpClientResponseHandler());
    }

    /**
    * Given a JSON response string, get a value for a specific key. Converts json {@literal <null>} to Java null.
    * @param responseBody The response to read
    * @param key The key to lookup
    * @return The value for the key, or null if not found
    */
    public static String getValueFromResponse(String responseBody, String key) throws JsonProcessingException {
        // TODO See if I can use OpenSearch XContent tools here to avoid Jackson dependency
        ObjectNode jsonResponse = (ObjectNode) objectMapper.readTree(responseBody);
        if (jsonResponse.has(key)) {
            if (jsonResponse.get(key).isNull()) {
                return null;
            }
            return jsonResponse.get(key).asText();
        }
        throw new IllegalArgumentException("Key " + key + " not found in response");
    }

    /**
     * Authenticate the HTTP request by manually setting the auth header.
     * This is favored over setting a global auth scheme to allow for dynamic credential updates.
     * @param request to be authenticated
     */
    public void authenticateRequest(HttpUriRequestBase request) {
        SecureString username = KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_SERVICE_USERNAME);
        SecureString password = KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_SERVICE_PASSWORD);

        if (password != null) {
            final String auth = username + ":" + password.clone();
            final byte[] encodedAuth = Base64.encodeBase64(auth.getBytes(StandardCharsets.ISO_8859_1));
            final String authHeader = "Basic " + new String(encodedAuth);
            request.setHeader(HttpHeaders.AUTHORIZATION, authHeader);
        }
    }

    /**
     * Close the httpClient
     */
    public void close() throws IOException {
        if (httpClient != null) {
            httpClient.close();
        }
    }
}
