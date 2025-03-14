/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.remoteindexbuild.client;

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
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.settings.SecureString;
import org.opensearch.index.IndexSettings;
import org.opensearch.remoteindexbuild.model.RemoteBuildRequest;
import org.opensearch.remoteindexbuild.model.RemoteBuildResponse;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.BASIC_PREFIX;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.BUCKET;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.BUILD_ENDPOINT;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.S3;

@SuppressWarnings("resource")
public class RemoteIndexHTTPClientTests extends OpenSearchSingleNodeTestCase {
    public static final String TEST_BUCKET = "test-bucket";
    public static final String TEST_CLUSTER = "test-cluster";
    public static final String MOCK_JOB_ID_RESPONSE = "{\"job_id\": \"job-1739930402\"}";
    public static final String MOCK_JOB_ID = "job-1739930402";
    public static final String MOCK_ENDPOINT = "https://mock-build-service.com";
    public static final String USERNAME = "username";
    public static final String PASSWORD = "password";

    @Mock
    protected static ClusterService clusterService;

    protected AutoCloseable openMocks;

    @Before
    public void setup() {
        openMocks = MockitoAnnotations.openMocks(this);
        clusterService = mock(ClusterService.class);
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
    }

    public void testGetAndCloseHttpclient_success() throws IOException {
        RemoteIndexHTTPClient client = new RemoteIndexHTTPClient(MOCK_ENDPOINT);
        assertNotNull(client);
        client.close();
    }

    public void testSubmitVectorBuild() throws IOException, URISyntaxException {
        CloseableHttpClient mockHttpClient = mock(CloseableHttpClient.class);
        try (MockedStatic<RemoteIndexHTTPClient> clientStaticMock = Mockito.mockStatic(RemoteIndexHTTPClient.class)) {
            clientStaticMock.when(RemoteIndexHTTPClient::getHttpClient).thenReturn(mockHttpClient);

            when(mockHttpClient.execute(any(HttpPost.class), any(HttpClientResponseHandler.class))).thenAnswer(
                response -> MOCK_JOB_ID_RESPONSE
            );

            MockedStatic<RemoteIndexHTTPClient> clientMock = Mockito.mockStatic(RemoteIndexHTTPClient.class);
            RemoteIndexHTTPClient client = new RemoteIndexHTTPClient(MOCK_ENDPOINT);
            clientMock.when(RemoteIndexHTTPClient::getHttpClient).thenReturn(mockHttpClient);
            RemoteBuildRequest mockBuildRequest = mock(RemoteBuildRequest.class);

            RemoteBuildResponse remoteBuildResponse = client.submitVectorBuild(mockBuildRequest);
            assertEquals(MOCK_JOB_ID, remoteBuildResponse.getJobId());

            ArgumentCaptor<HttpPost> requestCaptor = ArgumentCaptor.forClass(HttpPost.class);
            verify(mockHttpClient).execute(requestCaptor.capture(), any(HttpClientResponseHandler.class));
            HttpPost capturedRequest = requestCaptor.getValue();
            assertEquals(MOCK_ENDPOINT + BUILD_ENDPOINT, capturedRequest.getUri().toString());
            assertFalse(capturedRequest.containsHeader(HttpHeaders.AUTHORIZATION));
        }
    }

    public void testSecureSettingsReloadAndException() throws IOException, ProtocolException {
        CloseableHttpClient mockHttpClient = mock(CloseableHttpClient.class);
        try (MockedStatic<RemoteIndexHTTPClient> clientStaticMock = Mockito.mockStatic(RemoteIndexHTTPClient.class)) {
            clientStaticMock.when(RemoteIndexHTTPClient::getHttpClient).thenReturn(mockHttpClient);

            when(mockHttpClient.execute(any(HttpPost.class), any(HttpClientResponseHandler.class))).thenAnswer(
                response -> MOCK_JOB_ID_RESPONSE
            );

            RemoteIndexHTTPClient.reloadAuthHeader(new SecureString(USERNAME.toCharArray()), new SecureString(PASSWORD.toCharArray()));
            RemoteIndexHTTPClient client = new RemoteIndexHTTPClient(MOCK_ENDPOINT);

            ArgumentCaptor<HttpPost> requestCaptor = ArgumentCaptor.forClass(HttpPost.class);
            client.submitVectorBuild(mock(RemoteBuildRequest.class));

            verify(mockHttpClient).execute(requestCaptor.capture(), any(HttpClientResponseHandler.class));
            HttpPost capturedRequest = requestCaptor.getValue();
            Header authHeader = capturedRequest.getHeader(HttpHeaders.AUTHORIZATION);
            assertNotNull("Auth header should be set", authHeader);
            assertEquals(
                BASIC_PREFIX + Base64.encodeBase64String((USERNAME + ":" + PASSWORD).getBytes(StandardCharsets.ISO_8859_1)),
                authHeader.getValue()
            );
        }
        clearAuthHeader();
    }

    // Utility methods to populate settings for build requests

    public static RepositoryMetadata createTestRepositoryMetadata() {
        RepositoryMetadata metadata = mock(RepositoryMetadata.class);
        Settings repoSettings = Settings.builder().put(BUCKET, TEST_BUCKET).build();
        when(metadata.type()).thenReturn(S3);
        when(metadata.settings()).thenReturn(repoSettings);
        return metadata;
    }

    public static IndexSettings createTestIndexSettings() {
        IndexSettings mockIndexSettings = mock(IndexSettings.class);
        Settings indexSettingsSettings = Settings.builder().put(ClusterName.CLUSTER_NAME_SETTING.getKey(), TEST_CLUSTER).build();
        when(mockIndexSettings.getSettings()).thenReturn(indexSettingsSettings);
        return mockIndexSettings;
    }

    void clearAuthHeader() {
        RemoteIndexHTTPClient.reloadAuthHeader(null, null);
    }
}
