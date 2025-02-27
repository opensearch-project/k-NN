/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.core5.http.HttpHeaders;
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
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.blobstore.BlobStoreRepository;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING;
import static org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy.DOC_ID_FILE_EXTENSION;
import static org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy.VECTOR_BLOB_FILE_EXTENSION;

public class RemoteIndexHTTPClientTests extends OpenSearchSingleNodeTestCase {

    public static final String S3 = "s3";
    public static final String TEST_BUCKET = "test-bucket";
    public static final String BLOB = "blob";
    public static final String TEST_CLUSTER = "test-cluster";
    public static final String L2 = "l2";
    public static final String FP32 = "fp32";
    public static final String MOCK_JOB_ID_RESPONSE = "{\"job_id\": \"job-1739930402\"}";
    public static final String MOCK_JOB_ID = "job-1739930402";
    public static final String MOCK_BLOB_NAME = "blob";
    public static final String MOCK_ENDPOINT = "https://mock-build-service.com";
    @Mock
    protected ClusterService clusterService;

    protected AutoCloseable openMocks;

    private ObjectMapper mapper;

    @Before
    public void setup() {
        this.mapper = new ObjectMapper();
        openMocks = MockitoAnnotations.openMocks(this);
        clusterService = mock(ClusterService.class);
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        KNNSettings.state().setClusterService(clusterService);
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
    }

    public void testGetHttpClient_success() throws IOException {
        RemoteIndexHTTPClient client = RemoteIndexHTTPClient.getInstance();
        assertNotNull(client);
        client.close();
    }

    public void testConstructBuildRequestJson() throws IOException {
        Map<String, Object> algorithmParams = new HashMap<>();
        algorithmParams.put("ef_construction", 100);
        algorithmParams.put("m", 16);

        Map<String, Object> indexParameters = new HashMap<>();
        indexParameters.put("algorithm", "hnsw");
        indexParameters.put("space_type", "l2");
        indexParameters.put("algorithm_parameters", algorithmParams);

        RemoteBuildRequest request = RemoteBuildRequest.builder()
            .repositoryType("S3")
            .containerName("MyVectorStore")
            .vectorPath("MyVectorPath")
            .docIdPath("MyDocIdPath")
            .tenantId("MyTenant")
            .dimension(256)
            .docCount(1_000_000)
            .dataType("fp32")
            .engine("faiss")
            .indexParameters(indexParameters)
            .build();

        String expectedJson = "{"
            + "\"repository_type\":\"S3\","
            + "\"container_name\":\"MyVectorStore\","
            + "\"vector_path\":\"MyVectorPath\","
            + "\"doc_id_path\":\"MyDocIdPath\","
            + "\"tenant_id\":\"MyTenant\","
            + "\"dimension\":256,"
            + "\"doc_count\":1000000,"
            + "\"data_type\":\"fp32\","
            + "\"engine\":\"faiss\","
            + "\"index_parameters\":{"
            + "\"space_type\":\"l2\","
            + "\"algorithm\":\"hnsw\","
            + "\"algorithm_parameters\":{"
            + "\"ef_construction\":100,"
            + "\"m\":16"
            + "}"
            + "}"
            + "}";
        assertEquals(mapper.readTree(expectedJson), mapper.readTree(request.toJson()));
    }

    public void testGetValueFromResponse() throws JsonProcessingException {
        String jobID = "{\"job_id\": \"job-1739930402\"}";
        assertEquals("job-1739930402", RemoteIndexHTTPClient.getValueFromResponse(jobID, "job_id"));
        String failedIndexBuild = "{"
            + "\"task_status\":\"FAILED_INDEX_BUILD\","
            + "\"error\":\"Index build process interrupted.\","
            + "\"index_path\": null"
            + "}";
        String error = RemoteIndexHTTPClient.getValueFromResponse(failedIndexBuild, "error");
        assertEquals("Index build process interrupted.", error);
        assertNull(RemoteIndexHTTPClient.getValueFromResponse(failedIndexBuild, "index_path"));
    }

    public void testBuildRequest() {
        RepositoryMetadata metadata = mock(RepositoryMetadata.class);
        Settings repoSettings = Settings.builder().put("bucket", TEST_BUCKET).build();
        when(metadata.type()).thenReturn(S3);
        when(metadata.settings()).thenReturn(repoSettings);

        KNNSettings knnSettingsMock = mock(KNNSettings.class);
        IndexSettings mockIndexSettings = mock(IndexSettings.class);
        Settings indexSettingsSettings = Settings.builder().put(ClusterName.CLUSTER_NAME_SETTING.getKey(), TEST_CLUSTER).build();
        when(mockIndexSettings.getSettings()).thenReturn(indexSettingsSettings);

        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);

            List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
            final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
                vectorValues
            );
            final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
                VectorDataType.FLOAT,
                randomVectorValues
            );

            Map<String, Object> algorithmParams = Map.of("ef_construction", 94, "m", 2);

            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(
                    Map.of(
                        KNNConstants.SPACE_TYPE,
                        SpaceType.HAMMING.getValue(),
                        KNNConstants.NAME,
                        KNNConstants.METHOD_HNSW,
                        PARAMETERS,
                        algorithmParams
                    )
                )
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs(vectorValues.size())
                .build();

            RemoteBuildRequest request = RemoteIndexHTTPClient.getInstance()
                .constructBuildRequest(mockIndexSettings, buildIndexParams, metadata, "blob");

            assertEquals(S3, request.getRepositoryType());
            assertEquals(TEST_BUCKET, request.getContainerName());
            assertEquals(KNNConstants.FAISS_NAME, request.getEngine());
            assertEquals(FP32, request.getDataType());
            assertEquals(BLOB + VECTOR_BLOB_FILE_EXTENSION, request.getVectorPath());
            assertEquals(BLOB + DOC_ID_FILE_EXTENSION, request.getDocIdPath());
            assertEquals(TEST_CLUSTER, request.getTenantId());
            assertEquals(vectorValues.size(), request.getDocCount());
            assertEquals(2, request.getDimension());
            assertEquals(request.getIndexParameters().get(METHOD_PARAMETER_SPACE_TYPE), SpaceType.HAMMING.getValue());
            Object algorithmParameters = request.getIndexParameters().get("algorithm_parameters");
            Map<String, Object> algoMap = (Map<String, Object>) algorithmParameters;
            assertEquals(2, algoMap.get(METHOD_PARAMETER_M));
            assertEquals(94, algoMap.get(METHOD_PARAMETER_EF_CONSTRUCTION));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void testSubmitVectorBuild() throws IOException, URISyntaxException {
        CloseableHttpClient mockHttpClient = mock(CloseableHttpClient.class);
        RemoteIndexHTTPClient client = new RemoteIndexHTTPClient(mockHttpClient);

        when(mockHttpClient.execute(any(HttpPost.class), any(HttpClientResponseHandler.class))).thenAnswer(
            response -> MOCK_JOB_ID_RESPONSE
        );

        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        BlobStoreRepository blobStoreRepository = mock(BlobStoreRepository.class);
        RepositoryMetadata metadata = mock(RepositoryMetadata.class);
        Settings repoSettings = Settings.builder().put("bucket", TEST_BUCKET).build();

        when(metadata.type()).thenReturn(S3);
        when(metadata.settings()).thenReturn(repoSettings);
        when(blobStoreRepository.getMetadata()).thenReturn(metadata);
        when(repositoriesService.repository("test-repo")).thenReturn(blobStoreRepository);

        IndexSettings mockIndexSettings = mock(IndexSettings.class);
        Settings indexSettingsSettings = Settings.builder().put(ClusterName.CLUSTER_NAME_SETTING.getKey(), TEST_CLUSTER).build();
        when(mockIndexSettings.getSettings()).thenReturn(indexSettingsSettings);

        List<float[]> vectorValues = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            vectorValues
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);

        BuildIndexParams buildIndexParams = BuildIndexParams.builder()
            .knnEngine(KNNEngine.FAISS)
            .vectorDataType(VectorDataType.FLOAT)
            .parameters(Map.of(KNNConstants.SPACE_TYPE, L2, KNNConstants.NAME, KNNConstants.METHOD_HNSW))
            .knnVectorValuesSupplier(() -> knnVectorValues)
            .totalLiveDocs(vectorValues.size())
            .build();

        ClusterSettings clusterSettings = mock(ClusterSettings.class);
        when(clusterSettings.get(KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING)).thenReturn(MOCK_ENDPOINT);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);

        String jobId = client.submitVectorBuild(mockIndexSettings, buildIndexParams, metadata, MOCK_BLOB_NAME);
        // Isolated job_id from expectedResponse
        assertEquals(MOCK_JOB_ID, jobId);

        ArgumentCaptor<HttpPost> requestCaptor = ArgumentCaptor.forClass(HttpPost.class);
        Mockito.verify(mockHttpClient).execute(requestCaptor.capture(), any(HttpClientResponseHandler.class));
        HttpPost capturedRequest = requestCaptor.getValue();
        assertEquals(MOCK_ENDPOINT + RemoteIndexHTTPClient.BUILD_ENDPOINT, capturedRequest.getUri().toString());
        assert (!capturedRequest.containsHeader(HttpHeaders.AUTHORIZATION));
    }
}
