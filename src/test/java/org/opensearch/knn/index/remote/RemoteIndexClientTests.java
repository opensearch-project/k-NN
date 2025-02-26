/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Before;
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
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;

public class RemoteIndexClientTests extends KNNTestCase {

    public static final String S3 = "s3";
    public static final String TEST_BUCKET = "test-bucket";
    public static final String FAISS = "faiss";
    public static final String FLOAT = "float";
    public static final String BLOB_KNNVEC = "blob.knnvec";
    public static final String BLOB_KNNDID = "blob.knndid";
    public static final String TEST_CLUSTER = "test-cluster";
    public static final String L2 = "l2";
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

    public void testConstructBuildRequest() throws IOException {
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

    public void testBuildRequest() throws IOException {
        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        BlobStoreRepository blobStoreRepository = mock(BlobStoreRepository.class);
        RepositoryMetadata metadata = mock(RepositoryMetadata.class);
        Settings repoSettings = Settings.builder().put("bucket", "test-bucket").build();

        when(metadata.type()).thenReturn("s3");
        when(metadata.settings()).thenReturn(repoSettings);
        when(blobStoreRepository.getMetadata()).thenReturn(metadata);
        when(repositoriesService.repository("test-repo")).thenReturn(blobStoreRepository);

        KNNSettings knnSettingsMock = mock(KNNSettings.class);
        when(knnSettingsMock.getSettingValue(KNN_REMOTE_VECTOR_REPO_SETTING.getKey())).thenReturn("test-repo");

        IndexSettings mockIndexSettings = mock(IndexSettings.class);
        Settings indexSettingsSettings = Settings.builder().put(ClusterName.CLUSTER_NAME_SETTING.getKey(), "test-cluster").build();
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

            BuildIndexParams buildIndexParams = BuildIndexParams.builder()
                .knnEngine(KNNEngine.FAISS)
                .vectorDataType(VectorDataType.FLOAT)
                .parameters(Map.of(KNNConstants.SPACE_TYPE, L2))
                .knnVectorValuesSupplier(() -> knnVectorValues)
                .totalLiveDocs(vectorValues.size())
                .build();

            RemoteBuildRequest request = RemoteIndexHTTPClient.getInstance()
                .constructBuildRequest(mockIndexSettings, buildIndexParams, metadata, "blob");

            assertEquals(S3, request.getRepositoryType());
            assertEquals(TEST_BUCKET, request.getContainerName());
            assertEquals(FAISS, request.getEngine());
            assertEquals(FLOAT, request.getDataType()); // TODO this will be in {fp16, fp32, byte, binary}
            assertEquals(BLOB_KNNVEC, request.getVectorPath());
            assertEquals(BLOB_KNNDID, request.getDocIdPath());
            assertEquals(TEST_CLUSTER, request.getTenantId());
            assertEquals(vectorValues.size(), request.getDocCount());
        }
    }
}
