/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.client;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.StringUtils;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.client5.http.utils.Base64;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.HttpStatus;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.settings.SecureString;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.remoteindexbuild.model.RemoteBuildRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildResponse;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildStatusResponse;

import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;

import static org.apache.hc.core5.http.HttpStatus.SC_OK;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.BASIC_PREFIX;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.BUILD_ENDPOINT;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.JOB_ID_FIELD;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.STATUS_ENDPOINT;

/**
 * Class to handle all interactions with the remote vector build service.
 * Exceptions will cause a fallback to local CPU build.
 */
@Log4j2
public class RemoteIndexHTTPClient implements RemoteIndexClient, Closeable {
    private static volatile String authHeader = null;
    private final String endpoint;
    private final CloseableHttpClient httpClient;

    private static class HttpClientHolder {
        private static final CloseableHttpClient httpClient = HttpClients.custom()
            .setRetryStrategy(new RemoteIndexHTTPClientRetryStrategy())
            .build();
    }

    /**
     * Create the client, setting the endpoint per-instance
     */
    public RemoteIndexHTTPClient(final String endpoint) {
        this(endpoint, HttpClientHolder.httpClient);
    }

    /**
     * Create the client with a provided underlying HttpClient
     */
    RemoteIndexHTTPClient(final String endpoint, CloseableHttpClient client) {
        if (StringUtils.isEmpty(endpoint)) {
            throw new IllegalArgumentException("No endpoint set for RemoteIndexClient");
        }
        this.endpoint = endpoint;
        this.httpClient = client;
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
                (PrivilegedExceptionAction<String>) () -> httpClient.execute(buildRequest, body -> {
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
            RemoteBuildResponse buildResponse = RemoteBuildResponse.fromXContent(parser);
            if (StringUtils.isBlank(buildResponse.getJobId())) {
                throw new IOException("Invalid response format, empty " + JOB_ID_FIELD);
            }
            return buildResponse;
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
     * Helper method to directly get the status response for a given build
     * @param remoteBuildStatusRequest containing job ID to check
     * @return The entire response for the status request
     */
    public RemoteBuildStatusResponse getBuildStatus(RemoteBuildStatusRequest remoteBuildStatusRequest) throws IOException {
        String jobId = remoteBuildStatusRequest.getJobId();
        HttpGet request = new HttpGet(endpoint + STATUS_ENDPOINT + "/" + jobId);
        if (authHeader != null) {
            request.setHeader(HttpHeaders.AUTHORIZATION, authHeader);
        }
        try {
            String response = AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> httpClient.execute(request, body -> {
                if (body.getCode() < SC_OK || body.getCode() > HttpStatus.SC_MULTIPLE_CHOICES) {
                    throw new IOException("Failed to submit status request, got status code: " + body.getCode());
                }
                return EntityUtils.toString(body.getEntity());
            }));
            XContentParser parser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                LoggingDeprecationHandler.INSTANCE,
                response
            );
            return RemoteBuildStatusResponse.fromXContent(parser);
        } catch (Exception e) {
            throw new IOException("Failed to execute HTTP request", e);
        }
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
     * Set the global auth header for all HTTP requests.
     * @param username SecureString containing the username for basic authentication
     * @param password SecureString containing the password for basic authentication
     */
    public static void reloadAuthHeader(SecureString username, SecureString password) {
        if (password != null && password.length() > 0) {
            if (username == null || username.length() == 0) {
                throw new IllegalArgumentException("Username must be set if password is set");
            }
            final String auth = username + ":" + password.clone();
            final byte[] encodedAuth = Base64.encodeBase64(auth.getBytes(StandardCharsets.ISO_8859_1));
            authHeader = BASIC_PREFIX + new String(encodedAuth, StandardCharsets.UTF_8);
        } else {
            authHeader = null;
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
