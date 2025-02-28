/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.remote;

import org.apache.commons.codec.binary.Base64;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.core5.http.Header;
import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.ProtocolException;
import org.apache.hc.core5.http.io.HttpClientResponseHandler;
import org.junit.Before;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.MockSecureSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING;
import static org.opensearch.knn.index.SpaceType.L2;
import static org.opensearch.knn.index.VectorDataType.FLOAT;
import static org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy.DOC_ID_FILE_EXTENSION;
import static org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy.VECTOR_BLOB_FILE_EXTENSION;
import static org.opensearch.knn.index.engine.faiss.Faiss.getMFromIndexDescription;

public class RemoteIndexHTTPClientTests extends OpenSearchSingleNodeTestCase {

    public static final String S3 = "s3";
    public static final String TEST_BUCKET = "test-bucket";
    public static final String TEST_CLUSTER = "test-cluster";
    public static final String MOCK_JOB_ID_RESPONSE = "{\"job_id\": \"job-1739930402\"}";
    public static final String MOCK_JOB_ID = "job-1739930402";
    public static final String MOCK_BLOB_NAME = "blob";
    public static final String MOCK_ENDPOINT = "https://mock-build-service.com";
    public static final String USERNAME = "username";
    public static final String PASSWORD = "password";
    @Mock
    protected ClusterService clusterService;

    protected AutoCloseable openMocks;

    @Before
    public void setup() {
        openMocks = MockitoAnnotations.openMocks(this);
        clusterService = mock(ClusterService.class);
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        KNNSettings.state().setClusterService(clusterService);
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
    }

    public void testGetAndCloseHttpclient_success() throws IOException {
        setupTestClusterSettings();
        RemoteIndexHTTPClient client = new RemoteIndexHTTPClient();
        assertNotNull(client);
        client.close();
    }

    public void testGetValueFromResponse() throws IOException {
        String jobID = "{\"job_id\": \"job-1739930402\"}";
        assertEquals("job-1739930402", RemoteIndexHTTPClient.getValueFromResponse(jobID, JOB_ID));
        String failedIndexBuild = "{"
            + "\"task_status\":\"FAILED_INDEX_BUILD\","
            + "\"error_message\":\"Index build process interrupted.\","
            + "\"index_path\": null"
            + "}";
        String error = RemoteIndexHTTPClient.getValueFromResponse(failedIndexBuild, ERROR_MESSAGE);
        assertEquals("Index build process interrupted.", error);
        assertNull(RemoteIndexHTTPClient.getValueFromResponse(failedIndexBuild, INDEX_PATH));
    }

    public void testGetMFromIndexDescription() {
        assertEquals(16, getMFromIndexDescription("HNSW16,Flat"));
        assertEquals(8, getMFromIndexDescription("HNSW8,SQ"));
        assertThrows(IllegalArgumentException.class, () -> getMFromIndexDescription("Invalid description"));
    }

    public void testBuildRequest() {
        RepositoryMetadata metadata = createTestRepositoryMetadata();
        KNNSettings knnSettingsMock = mock(KNNSettings.class);
        IndexSettings mockIndexSettings = createTestIndexSettings();
        setupTestClusterSettings();

        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);
            when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING.getKey())).thenReturn(MOCK_ENDPOINT);
            KNNSettings.state().setClusterService(clusterService);

            BuildIndexParams indexInfo = createTestBuildIndexParams();

            HTTPRemoteBuildRequest request = RemoteBuildRequestBuilder.builder(HTTPRemoteBuildRequest.class)
                .indexSettings(mockIndexSettings)
                .indexInfo(indexInfo)
                .repositoryMetadata(metadata)
                .blobName(MOCK_BLOB_NAME)
                .build();

            assertEquals(S3, request.getRepositoryType());
            assertEquals(TEST_BUCKET, request.getContainerName());
            assertEquals(FAISS_NAME, request.getEngine());
            assertEquals(FLOAT.getValue(), request.getVectorDataType());
            assertEquals(MOCK_BLOB_NAME + VECTOR_BLOB_FILE_EXTENSION, request.getVectorPath());
            assertEquals(MOCK_BLOB_NAME + DOC_ID_FILE_EXTENSION, request.getDocIdPath());
            assertEquals(TEST_CLUSTER, request.getTenantId());
            assertEquals(2, request.getDocCount());
            assertEquals(2, request.getDimension());

            String expectedJson = "{"
                + "\"repository_type\":\"s3\","
                + "\"container_name\":\"test-bucket\","
                + "\"vector_path\":\"blob.knnvec\","
                + "\"doc_id_path\":\"blob.knndid\","
                + "\"tenant_id\":\"test-cluster\","
                + "\"dimension\":2,"
                + "\"doc_count\":2,"
                + "\"data_type\":\"float\","
                + "\"engine\":\"faiss\","
                + "\"index_parameters\":{"
                + "\"space_type\":\"l2\","
                + "\"algorithm\":\"hnsw\","
                + "\"algorithm_parameters\":{"
                + "\"ef_construction\":94,"
                + "\"ef_search\":89,"
                + "\"m\":14"
                + "}"
                + "}"
                + "}";
            XContentParser parser1 = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                expectedJson
            );

            XContentParser parser2 = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                request.toJson()
            );

            assertEquals(parser1.map(), parser2.map());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void testSubmitVectorBuild() throws IOException, URISyntaxException {
        RepositoryMetadata metadata = createTestRepositoryMetadata();
        KNNSettings knnSettingsMock = mock(KNNSettings.class);
        IndexSettings mockIndexSettings = createTestIndexSettings();
        setupTestClusterSettings();

        CloseableHttpClient mockHttpClient = mock(CloseableHttpClient.class);
        try (MockedStatic<RemoteIndexHTTPClient> clientStaticMock = Mockito.mockStatic(RemoteIndexHTTPClient.class)) {
            clientStaticMock.when(RemoteIndexHTTPClient::getHttpClient).thenReturn(mockHttpClient);

            try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
                knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);
                when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING.getKey())).thenReturn(MOCK_ENDPOINT);
                KNNSettings.state().setClusterService(clusterService);

                BuildIndexParams buildIndexParams = createTestBuildIndexParams();

                when(mockHttpClient.execute(any(HttpPost.class), any(HttpClientResponseHandler.class))).thenAnswer(
                    response -> MOCK_JOB_ID_RESPONSE
                );

                RemoteIndexHTTPClient client = new RemoteIndexHTTPClient();
                clientStaticMock.when(() -> RemoteIndexHTTPClient.getValueFromResponse(any(String.class), any(String.class)))
                    .thenCallRealMethod();

                RemoteBuildResponse remoteBuildResponse = client.submitVectorBuild(
                    new HTTPRemoteBuildRequest(mockIndexSettings, buildIndexParams, metadata, MOCK_BLOB_NAME)
                );
                assertEquals(MOCK_JOB_ID, remoteBuildResponse.jobId());

                ArgumentCaptor<HttpPost> requestCaptor = ArgumentCaptor.forClass(HttpPost.class);
                verify(mockHttpClient).execute(requestCaptor.capture(), any(HttpClientResponseHandler.class));
                HttpPost capturedRequest = requestCaptor.getValue();
                assertEquals(MOCK_ENDPOINT + BUILD_ENDPOINT, capturedRequest.getUri().toString());
                assert (!capturedRequest.containsHeader(HttpHeaders.AUTHORIZATION));
            }
        }
    }

    public void testSecureSettingsReloadAndException() throws IOException {
        final MockSecureSettings secureSettings = new MockSecureSettings();
        secureSettings.setString(KNN_REMOTE_BUILD_CLIENT_USERNAME_SETTING.getKey(), USERNAME);
        secureSettings.setString(KNN_REMOTE_BUILD_CLIENT_PASSWORD_SETTING.getKey(), PASSWORD);
        final Settings settings = Settings.builder().setSecureSettings(secureSettings).build();

        CloseableHttpClient mockHttpClient = mock(CloseableHttpClient.class);
        try (MockedStatic<RemoteIndexHTTPClient> clientStaticMock = Mockito.mockStatic(RemoteIndexHTTPClient.class)) {
            clientStaticMock.when(RemoteIndexHTTPClient::getHttpClient).thenReturn(mockHttpClient);

            try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
                KNNSettings knnSettingsMock = mock(KNNSettings.class);
                knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);
                when(knnSettingsMock.getSettingValue(KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING.getKey())).thenReturn(MOCK_ENDPOINT);
                KNNSettings.state().setClusterService(clusterService);

                when(mockHttpClient.execute(any(HttpPost.class), any(HttpClientResponseHandler.class))).thenAnswer(
                    response -> MOCK_JOB_ID_RESPONSE
                );

                RepositoryMetadata metadata = createTestRepositoryMetadata();
                IndexSettings mockIndexSettings = createTestIndexSettings();
                setupTestClusterSettings();
                BuildIndexParams buildIndexParams = createTestBuildIndexParams();

                clientStaticMock.when(() -> RemoteIndexHTTPClient.reloadAuthHeader(any(Settings.class))).thenCallRealMethod();
                clientStaticMock.when(() -> RemoteIndexHTTPClient.getValueFromResponse(any(String.class), any(String.class)))
                    .thenCallRealMethod();

                RemoteIndexHTTPClient client = new RemoteIndexHTTPClient();
                RemoteIndexHTTPClient.reloadAuthHeader(settings);

                ArgumentCaptor<HttpPost> requestCaptor = ArgumentCaptor.forClass(HttpPost.class);
                client.submitVectorBuild(new HTTPRemoteBuildRequest(mockIndexSettings, buildIndexParams, metadata, MOCK_BLOB_NAME));

                verify(mockHttpClient).execute(requestCaptor.capture(), any(HttpClientResponseHandler.class));
                HttpPost capturedRequest = requestCaptor.getValue();
                Header authHeader = capturedRequest.getHeader(HttpHeaders.AUTHORIZATION);
                assertNotNull("Auth header should be set", authHeader);
                assertEquals(
                    "Basic " + Base64.encodeBase64String((USERNAME + ":" + PASSWORD).getBytes(StandardCharsets.ISO_8859_1)),
                    authHeader.getValue()
                );
            } catch (ProtocolException e) {
                throw new RuntimeException(e);
            }
        }
    }

    // Utility methods to populate settings for build requests

    private BuildIndexParams createTestBuildIndexParams() {
        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(FLOAT, randomVectorValues);

        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(NAME, ENCODER_FLAT);
        encoderParams.put(PARAMETERS, Map.of());

        Map<String, Object> algorithmParams = new HashMap<>();
        algorithmParams.put(METHOD_PARAMETER_EF_SEARCH, 89);
        algorithmParams.put(METHOD_PARAMETER_EF_CONSTRUCTION, 94);
        algorithmParams.put(ENCODER_FLAT, encoderParams);

        Map<String, Object> parameters = new HashMap<>();
        parameters.put(NAME, METHOD_HNSW);
        parameters.put(VECTOR_DATA_TYPE_FIELD, FLOAT.getValue());
        parameters.put(INDEX_DESCRIPTION_PARAMETER, "HNSW14,Flat");
        parameters.put(SPACE_TYPE, L2.getValue());
        parameters.put(PARAMETERS, algorithmParams);

        return BuildIndexParams.builder()
            .knnEngine(KNNEngine.FAISS)
            .vectorDataType(FLOAT)
            .parameters(parameters)
            .knnVectorValuesSupplier(() -> knnVectorValues)
            .totalLiveDocs(vectorValues.size())
            .build();
    }

    private RepositoryMetadata createTestRepositoryMetadata() {
        RepositoryMetadata metadata = mock(RepositoryMetadata.class);
        Settings repoSettings = Settings.builder().put(BUCKET, TEST_BUCKET).build();
        when(metadata.type()).thenReturn(S3);
        when(metadata.settings()).thenReturn(repoSettings);
        return metadata;
    }

    private IndexSettings createTestIndexSettings() {
        IndexSettings mockIndexSettings = mock(IndexSettings.class);
        Settings indexSettingsSettings = Settings.builder().put(ClusterName.CLUSTER_NAME_SETTING.getKey(), TEST_CLUSTER).build();
        when(mockIndexSettings.getSettings()).thenReturn(indexSettingsSettings);
        return mockIndexSettings;
    }

    private void setupTestClusterSettings() {
        ClusterSettings clusterSettings = mock(ClusterSettings.class);
        when(clusterSettings.get(KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING)).thenReturn(MOCK_ENDPOINT);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
    }
}
