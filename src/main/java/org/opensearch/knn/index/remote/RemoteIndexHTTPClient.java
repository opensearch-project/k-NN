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
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.settings.SecureString;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.plugin.KNNPlugin;

import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;

import static org.apache.hc.core5.http.HttpStatus.SC_OK;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.BUILD_ENDPOINT;

/**
 * Class to handle all interactions with the remote vector build service.
 * Exceptions will cause a fallback to local CPU build.
 */
@Log4j2
public class RemoteIndexHTTPClient implements RemoteIndexClient, Closeable {
    private static final String BASIC_PREFIX = "Basic ";

    private static volatile String authHeader = null;

    private final String endpoint;

    private static class HttpClientHolder {
        private static final CloseableHttpClient httpClient = createHttpClient();

        private static CloseableHttpClient createHttpClient() {
            return HttpClients.custom().setRetryStrategy(new RemoteIndexHTTPClientRetryStrategy()).build();
        }
    }

    /**
     * Get the Singleton shared HTTP client
     * @return The static HTTP Client
     */
    protected static CloseableHttpClient getHttpClient() {
        return HttpClientHolder.httpClient;
    }

    /**
     * Creates the client, setting the endpoint per-instance so the same endpoint is used per-build operation
     * (per call to {@link RemoteIndexBuildStrategy#buildAndWriteIndex(BuildIndexParams)})
     */
    public RemoteIndexHTTPClient() {
        String endpoint = KNNSettings.getRemoteBuildServiceEndpoint();
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
        HttpPost buildRequest = getHttpPost(toJson(remoteBuildRequest));
        try {
            String response = AccessController.doPrivileged(
                (PrivilegedExceptionAction<String>) () -> getHttpClient().execute(buildRequest, body -> {
                    if (body.getCode() < SC_OK || body.getCode() > HttpStatus.SC_MULTIPLE_CHOICES) {
                        throw new IOException("Failed to submit build request, got status code: " + body.getCode());
                    }
                    return EntityUtils.toString(body.getEntity());
                })
            );
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                LoggingDeprecationHandler.INSTANCE,
                response
            );
            return RemoteBuildResponse.fromXContent(parser);
        } catch (Exception e) {
            throw new IOException("Failed to execute HTTP request", e);
        }
    }

    /**
     * Helper method to form the HttpPost request from the HTTPRemoteBuildRequest
     * @param jsonRequest JSON converted request body to be submitted
     * @return HttpPost request to be submitted
     */
    private HttpPost getHttpPost(String jsonRequest) {
        HttpPost buildRequest = new HttpPost(URI.create(endpoint) + BUILD_ENDPOINT);
        buildRequest.setHeader(HttpHeaders.CONTENT_TYPE, ContentType.APPLICATION_JSON.toString());
        buildRequest.setEntity(new StringEntity(jsonRequest, ContentType.APPLICATION_JSON));
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
     * Convert the RemoteBuildRequest object to a JSON object for this specific HTTP implementation.
     * @param object RemoteBuildRequest with parameters
     * @return JSON String representation of the request body
     * @throws IOException if the request cannot be converted to JSON
     */
    private String toJson(ToXContentObject object) throws IOException {
        try (XContentBuilder builder = JsonXContent.contentBuilder()) {
            object.toXContent(builder, ToXContentObject.EMPTY_PARAMS);
            return builder.toString();
        }
    }

    /**
     * Set the global auth header to use the refreshed secure settings.
     * Called by {@link KNNPlugin#reload(Settings)} when the nodes reload API is called.
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
