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
import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.http.io.entity.StringEntity;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.settings.SecureString;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_DEFAULT_SPACE_TYPE;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING;

/**
 * Class to handle all interactions with the remote vector build service.
 * InterruptedExceptions will cause a fallback to local CPU build.
 */
@Log4j2
public class RemoteIndexHTTPClient implements RemoteIndexClient, Closeable {
    private static RemoteIndexHTTPClient INSTANCE;
    private volatile CloseableHttpClient httpClient;
    protected static final int MAX_RETRIES = 1; // 2 total attempts
    protected static final long BASE_DELAY_MS = 1000;
    protected static final String BUILD_ENDPOINT = "/_build";

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
    * @return job_id from the server response used to track the job
    */
    @Override
    public String submitVectorBuild(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String blobName
    ) throws IOException {
        RemoteBuildRequest request = constructBuildRequest(indexSettings, indexInfo, repositoryMetadata, blobName);
        URI endpoint = URI.create(KNNSettings.state().getSettingValue(KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING.getKey()));
        HttpPost buildRequest = new HttpPost(endpoint + BUILD_ENDPOINT);
        buildRequest.setHeader("Content-Type", "application/json");
        buildRequest.setEntity(new StringEntity(request.toJson()));
        if (authHeader != null) {
            buildRequest.setHeader(HttpHeaders.AUTHORIZATION, authHeader);
        }

        try {
            String response = AccessController.doPrivileged(
                (PrivilegedExceptionAction<String>) () -> httpClient.execute(buildRequest, body -> {
                    if (body.getCode() != 200) {
                        throw new IOException("Failed to submit build request, got status code: " + body.getCode());
                    }
                    return EntityUtils.toString(body.getEntity());
                })
            );

            if (response == null) {
                throw new IOException("Received 200 status code but response is null.");
            }

            return getValueFromResponse(response, "job_id");
        } catch (PrivilegedActionException e) {
            throw new IOException("Failed to execute HTTP request", e.getException());
        }
    }

    /**
    * Await the completion of the index build by polling periodically and handling the returned statuses until timeout.
    * @param jobId identifier from the server to track the job
    * @return the path to the completed index
    */
    @Override
    public String awaitVectorBuild(String jobId) {
        throw new NotImplementedException();
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
     * Construct the RemoteBuildRequest object for the index build request
     * @param indexSettings Index settings
     * @param indexInfo Index parameters
     * @param repositoryMetadata Metadata of the repository containing the index
     * @param blobName File name generated by the Build Strategy with a UUID
     * @return RemoteBuildRequest with parameters set
     */
    RemoteBuildRequest constructBuildRequest(
        IndexSettings indexSettings,
        BuildIndexParams indexInfo,
        RepositoryMetadata repositoryMetadata,
        String blobName
    ) throws IOException {
        String repositoryType = repositoryMetadata.type();
        String containerName;
        switch (repositoryType) {
            case "s3" -> containerName = repositoryMetadata.settings().get("bucket");
            default -> throw new IllegalArgumentException(
                "Repository type " + repositoryType + " is not supported by the remote build service"
            );
        }
        String vectorDataType = indexInfo.getVectorDataType().getValue();

        KNNVectorValues<?> vectorValues = indexInfo.getKnnVectorValuesSupplier().get();
        KNNCodecUtil.initializeVectorValues(vectorValues);
        assert (vectorValues.dimension() > 0);

        Map<String, Object> indexParameters = null;
        if (indexInfo.getParameters() != null) {
            indexParameters = constructIndexParams(indexInfo);
        }

        return RemoteBuildRequest.builder()
            .repositoryType(repositoryType)
            .containerName(containerName)
            .vectorPath(blobName + RemoteIndexBuildStrategy.VECTOR_BLOB_FILE_EXTENSION)
            .docIdPath(blobName + RemoteIndexBuildStrategy.DOC_ID_FILE_EXTENSION)
            .tenantId(indexSettings.getSettings().get(ClusterName.CLUSTER_NAME_SETTING.getKey()))
            .dimension(vectorValues.dimension())
            .docCount(indexInfo.getTotalLiveDocs())
            .dataType(vectorDataType)
            .engine(indexInfo.getKnnEngine().getName())
            .indexParameters(indexParameters)
            .build();
    }

    /**
     * Helper method to construct the index parameter object. Depending on the engine and algorithm, different parameters are needed.
     * @param indexInfo Index parameters
     * @return Map of necessary index parameters
     */
    private Map<String, Object> constructIndexParams(BuildIndexParams indexInfo) {
        Map<String, Object> indexParameters = new HashMap<>();
        String methodName = (String) indexInfo.getParameters().get(NAME);
        indexParameters.put("algorithm", methodName);
        indexParameters.put(METHOD_PARAMETER_SPACE_TYPE, indexInfo.getParameters().getOrDefault(SPACE_TYPE, INDEX_KNN_DEFAULT_SPACE_TYPE));

        if (indexInfo.getParameters().containsKey(PARAMETERS)) {
            Object innerParams = indexInfo.getParameters().get(PARAMETERS);
            if (innerParams instanceof Map) {
                Map<String, Object> algorithmParams = new HashMap<>();
                Map<String, Object> innerMap = (Map<String, Object>) innerParams;
                switch (methodName) {
                    case METHOD_HNSW -> {
                        algorithmParams.put(
                            METHOD_PARAMETER_EF_CONSTRUCTION,
                            innerMap.getOrDefault(METHOD_PARAMETER_EF_CONSTRUCTION, INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION)
                        );
                        algorithmParams.put(METHOD_PARAMETER_M, innerMap.getOrDefault(METHOD_PARAMETER_M, INDEX_KNN_DEFAULT_ALGO_PARAM_M));
                        if (indexInfo.getKnnEngine().getName().equals(FAISS_NAME)) {
                            algorithmParams.put(
                                METHOD_PARAMETER_EF_SEARCH,
                                innerMap.getOrDefault(METHOD_PARAMETER_EF_SEARCH, INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH)
                            );
                        }
                    }
                    case METHOD_IVF -> {
                        algorithmParams.put(
                            METHOD_PARAMETER_NLIST,
                            innerMap.getOrDefault(METHOD_PARAMETER_NLIST, METHOD_PARAMETER_NLIST_DEFAULT)
                        );
                        algorithmParams.put(
                            METHOD_PARAMETER_NPROBES,
                            innerMap.getOrDefault(METHOD_PARAMETER_NPROBES, METHOD_PARAMETER_NPROBES_DEFAULT)
                        );
                    }
                }
                indexParameters.put("algorithm_parameters", algorithmParams);
            }
        }
        return indexParameters;
    }

    /**
     * Close the httpClient
     */
    public void close() throws IOException {
        if (httpClient != null) {
            httpClient.close();
        }
    }

    /**
     * Rebuild the httpClient with the new credentials
     * @param settings Settings to use to get the new credentials
     */
    public void reloadSecureSettings(Settings settings) {
        SecureString username = KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING.get(settings);
        SecureString password = KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING.get(settings);

        if (password != null && !password.isEmpty()) {
            if (username == null || username.isEmpty()) {
                throw new IllegalArgumentException("Username must be set if password is set");
            }
            final String auth = username + ":" + password.clone();
            final byte[] encodedAuth = Base64.encodeBase64(auth.getBytes(StandardCharsets.ISO_8859_1));
            this.authHeader = "Basic " + new String(encodedAuth);
        } else {
            this.authHeader = null;
        }
    }
}
