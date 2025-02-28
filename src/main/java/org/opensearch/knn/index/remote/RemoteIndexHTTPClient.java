/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.NotImplementedException;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.client5.http.utils.Base64;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.HttpStatus;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.settings.SecureString;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteStatusResponse;

import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.Map;

import static org.apache.hc.core5.http.HttpStatus.SC_OK;
import static org.opensearch.knn.common.KNNConstants.JOB_ID;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING;

/**
 * Class to handle all interactions with the remote vector build service.
 * Exceptions will cause a fallback to local CPU build.
 */
@Log4j2
public class RemoteIndexHTTPClient implements RemoteIndexClient, Closeable {
    public static final String BASIC_PREFIX = "Basic ";
    private static volatile String authHeader = null;

    private final String endpoint;

    private static class HttpClientHolder {
        private static final CloseableHttpClient httpClient = createHttpClient();

        private static CloseableHttpClient createHttpClient() {
            return HttpClients.custom().setRetryStrategy(new RemoteIndexClientRetryStrategy()).build();
        }
    }

    /**
     * Get the Singleton shared HTTP client
     * @return The static HTTP Client
     */
    protected static CloseableHttpClient getHttpClient() {
        return HttpClientHolder.httpClient;
    }

    public RemoteIndexHTTPClient() {
        String endpoint = KNNSettings.state().getSettingValue(KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING.getKey());
        if (endpoint == null || endpoint.isEmpty()) {
            throw new IllegalArgumentException("No endpoint set for RemoteIndexClient");
        }
        this.endpoint = endpoint;
    }

    /**
    * Submit a build to the Remote Vector Build Service endpoint.
    * @return RemoteBuildResponse containing job_id from the server response used to track the job
    */
    @Override
    public RemoteBuildResponse submitVectorBuild(RemoteBuildRequest remoteBuildRequest) throws IOException {
        assert (remoteBuildRequest instanceof HTTPRemoteBuildRequest);
        HTTPRemoteBuildRequest request = (HTTPRemoteBuildRequest) remoteBuildRequest;
        HttpPost buildRequest = getHttpPost(request);

        try {
            String response = AccessController.doPrivileged(
                (PrivilegedExceptionAction<String>) () -> getHttpClient().execute(buildRequest, body -> {
                    if (body.getCode() < SC_OK || body.getCode() > HttpStatus.SC_MULTIPLE_CHOICES) {
                        throw new IOException("Failed to submit build request, got status code: " + body.getCode());
                    }
                    return EntityUtils.toString(body.getEntity());
                })
            );

            if (response == null || response.isEmpty()) {
                throw new IOException("Received success status code but response is null or empty.");
            }
            String jobId = getValueFromResponse(response, JOB_ID);
            if (jobId == null || jobId.isEmpty()) {
                throw new IOException("Received success status code but " + JOB_ID + " is null or empty.");
            }
            return new RemoteBuildResponse(jobId);
        } catch (Exception e) {
            throw new IOException("Failed to execute HTTP request", e);
        }
    }

    /**
     * Helper method to form the HttpPost request from the HTTPRemoteBuildRequest
     * @param request HTTPRemoteBuildRequest to be submitted
     * @return HttpPost request to be submitted
     * @throws IOException if the request cannot be formed
     */
    private HttpPost getHttpPost(HTTPRemoteBuildRequest request) throws IOException {
        HttpPost buildRequest = new HttpPost(URI.create(endpoint) + KNNConstants.BUILD_ENDPOINT);
        buildRequest.setHeader(HttpHeaders.CONTENT_TYPE, ContentType.APPLICATION_JSON.toString());
        buildRequest.setEntity(new StringEntity(request.toJson()));
        if (authHeader != null) {
            buildRequest.setHeader(HttpHeaders.AUTHORIZATION, authHeader);
        }
        return buildRequest;
    }

    /**
    * Await the completion of the index build by polling periodically and handling the returned statuses until timeout.
    * @param remoteBuildResponse containing job_id from the server response used to track the job
    * @return RemoteStatusResponse containing the path to the completed index
    */
    @Override
    public RemoteStatusResponse awaitVectorBuild(RemoteBuildResponse remoteBuildResponse) {
        throw new NotImplementedException();
    }

    /**
    * Given a JSON response string, get a value for a specific key. Converts json {@literal <null>} to Java null.
    * @param responseBody The response to read
    * @param key The key to lookup
    * @return The value for the key
    */
    static String getValueFromResponse(String responseBody, String key) throws IOException {
        try (
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                responseBody
            )
        ) {
            Map<String, Object> responseMap = parser.map();
            if (responseMap.containsKey(key)) {
                Object value = responseMap.get(key);
                if (value == null) {
                    return null;
                }
                return value.toString();
            }
            throw new IllegalArgumentException("Key " + key + " not found in response");
        }
    }

    /**
     * Set the global auth header to use the refreshed secure settings
     * @param settings Settings to use to get the credentials
     */
    public static void reloadAuthHeader(Settings settings) {
        SecureString username = KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING.get(settings);
        SecureString password = KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING.get(settings);

        if (password != null && !password.isEmpty()) {
            if (username == null || username.isEmpty()) {
                throw new IllegalArgumentException("Username must be set if password is set");
            }
            final String auth = username + ":" + password.clone();
            final byte[] encodedAuth = Base64.encodeBase64(auth.getBytes(StandardCharsets.ISO_8859_1));
            authHeader = BASIC_PREFIX + new String(encodedAuth);
        } else {
            authHeader = null;
        }
    }

    /**
     * Close the httpClient
     */
    public void close() throws IOException {
        if (getHttpClient() != null) {
            getHttpClient().close();
        }
    }
}
