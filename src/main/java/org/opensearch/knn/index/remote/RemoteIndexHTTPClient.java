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
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.client5.http.utils.Base64;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.HttpStatus;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.settings.SecureString;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteStatusResponse;

import java.io.Closeable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;

import static org.apache.hc.core5.http.HttpStatus.SC_OK;
import static org.opensearch.knn.common.KNNConstants.JOB_ID;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING;

/**
 * Class to handle all interactions with the remote vector build service.
 * InterruptedExceptions will cause a fallback to local CPU build.
 */
@Log4j2
public class RemoteIndexHTTPClient implements RemoteIndexClient, Closeable {
    public static final String BASIC_PREFIX = "Basic ";
    private static RemoteIndexHTTPClient INSTANCE;
    private volatile CloseableHttpClient httpClient;

    private static final ObjectMapper objectMapper = new ObjectMapper();
    private String authHeader = null;

    /**
     * Return the Singleton instance of the node's RemoteIndexClient
     * @return RemoteIndexClient instance
     */
    public static synchronized RemoteIndexHTTPClient getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new RemoteIndexHTTPClient(createHttpClient());
        }
        return INSTANCE;
    }

    /**
     * Initialize the httpClient to be used
     * @return The HTTP Client
     */
    private static CloseableHttpClient createHttpClient() {
        return HttpClients.custom().setRetryStrategy(new RemoteIndexClientRetryStrategy()).build();
    }

    RemoteIndexHTTPClient(CloseableHttpClient httpClient) {
        this.httpClient = httpClient;
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
                (PrivilegedExceptionAction<String>) () -> httpClient.execute(buildRequest, body -> {
                    if (body.getCode() < SC_OK || body.getCode() > HttpStatus.SC_MULTIPLE_CHOICES) {
                        throw new IOException("Failed to submit build request, got status code: " + body.getCode());
                    }
                    return EntityUtils.toString(body.getEntity());
                })
            );

            if (response == null) {
                throw new IOException("Received success status code but response is null.");
            }

            return new HTTPRemoteBuildResponse(getValueFromResponse(response, JOB_ID), request.getEndpoint());
        } catch (Exception e) {
            throw new IOException("Failed to execute HTTP request", e);
        }
    }

    private HttpPost getHttpPost(HTTPRemoteBuildRequest request) throws IOException {
        HttpPost buildRequest = new HttpPost(request.getEndpoint() + KNNConstants.BUILD_ENDPOINT);
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
     * Construct the HTTP specific build request (with endpoint and .toJson method)
     * @param indexSettings IndexSettings for the index being built
     * @param indexInfo BuildIndexParams for the index being built
     * @param repositoryMetadata Metadata of the vector repository
     * @param blobName The name of the blob written to the repo, to be suffixed with ".knnvec" or ".knndid"
     * @return RemoteBuildRequest with parameters set
     */
    @Override
    public RemoteBuildRequest constructBuildRequest(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String blobName
    ) throws IOException {
        return new HTTPRemoteBuildRequest(indexSettings, indexInfo, repositoryMetadata, blobName);
    }

    /**
    * Given a JSON response string, get a value for a specific key. Converts json {@literal <null>} to Java null.
    * @param responseBody The response to read
    * @param key The key to lookup
    * @return The value for the key, or null if not found
    */
    static String getValueFromResponse(String responseBody, String key) throws JsonProcessingException {
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
     * Set the global auth header to use the refreshed secure settings
     * @param settings Settings to use to get the credentials
     */
    public void reloadAuthHeader(Settings settings) {
        SecureString username = KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING.get(settings);
        SecureString password = KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING.get(settings);

        if (password != null && !password.isEmpty()) {
            if (username == null || username.isEmpty()) {
                throw new IllegalArgumentException("Username must be set if password is set");
            }
            final String auth = username + ":" + password.clone();
            final byte[] encodedAuth = Base64.encodeBase64(auth.getBytes(StandardCharsets.ISO_8859_1));
            this.authHeader = BASIC_PREFIX + new String(encodedAuth);
        } else {
            this.authHeader = null;
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
