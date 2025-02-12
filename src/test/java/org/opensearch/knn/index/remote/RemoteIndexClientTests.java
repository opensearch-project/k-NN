/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class RemoteIndexClientTests extends OpenSearchSingleNodeTestCase {

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
        RemoteIndexClient client = RemoteIndexClient.getInstance();
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
        assertEquals("job-1739930402", RemoteIndexClient.getValueFromResponse(jobID, "job_id"));
        String failedIndexBuild = "{"
            + "\"task_status\":\"FAILED_INDEX_BUILD\","
            + "\"error\":\"Index build process interrupted.\","
            + "\"index_path\": null"
            + "}";
        String error = RemoteIndexClient.getValueFromResponse(failedIndexBuild, "error");
        assertEquals("Index build process interrupted.", error);
        assertNull(RemoteIndexClient.getValueFromResponse(failedIndexBuild, "index_path"));
    }
}
