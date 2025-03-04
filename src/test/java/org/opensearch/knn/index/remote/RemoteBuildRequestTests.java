/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.remote;

import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.index.VectorDataType.FLOAT;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.DOC_ID_FILE_EXTENSION;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.S3;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.VECTOR_BLOB_FILE_EXTENSION;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.MOCK_BLOB_NAME;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.TEST_BUCKET;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.TEST_CLUSTER;

public class RemoteBuildRequestTests extends OpenSearchSingleNodeTestCase {
    @Mock
    protected static ClusterService clusterService;

    protected AutoCloseable openMocks;

    @Before
    public void setup() {
        openMocks = MockitoAnnotations.openMocks(this);
        clusterService = mock(ClusterService.class);
        Set<Setting<?>> defaultClusterSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        KNNSettings.state().setClusterService(clusterService);
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(Settings.EMPTY, defaultClusterSettings));
    }

    /**
     * Test the construction of the build request by comparing it to an explicitly created JSON object.
     */
    public void testBuildRequest() {
        RepositoryMetadata metadata = RemoteIndexHTTPClientTests.createTestRepositoryMetadata();
        KNNSettings knnSettingsMock = mock(KNNSettings.class);
        IndexSettings mockIndexSettings = RemoteIndexHTTPClientTests.createTestIndexSettings();

        try (MockedStatic<KNNSettings> knnSettingsStaticMock = Mockito.mockStatic(KNNSettings.class)) {
            knnSettingsStaticMock.when(KNNSettings::state).thenReturn(knnSettingsMock);
            KNNSettings.state().setClusterService(clusterService);

            BuildIndexParams indexInfo = RemoteIndexHTTPClientTests.createTestBuildIndexParams();

            RemoteBuildRequest request = new RemoteBuildRequest(mockIndexSettings, indexInfo, metadata, MOCK_BLOB_NAME);

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
            XContentParser expectedParser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                expectedJson
            );
            Map<String, Object> expectedMap = expectedParser.map();

            String jsonRequest;
            try (XContentBuilder builder = JsonXContent.contentBuilder()) {
                request.toXContent(builder, ToXContentObject.EMPTY_PARAMS);
                jsonRequest = builder.toString();
            }

            XContentParser generatedParser = JsonXContent.jsonXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                jsonRequest
            );
            Map<String, Object> generatedMap = generatedParser.map();

            assertEquals(expectedMap, generatedMap);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
