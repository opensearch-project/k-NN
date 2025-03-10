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
import org.opensearch.common.blobstore.BlobPath;
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
import org.opensearch.repositories.blobstore.BlobStoreRepository;
import org.opensearch.test.OpenSearchSingleNodeTestCase;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.SpaceType.L2;
import static org.opensearch.knn.index.VectorDataType.FLOAT;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.ALGORITHM;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.ALGORITHM_PARAMETERS;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.CONTAINER_NAME;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.DOC_COUNT;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.DOC_ID_FILE_EXTENSION;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.DOC_ID_PATH;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.INDEX_PARAMETERS;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.REPOSITORY_TYPE;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.S3;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.TENANT_ID;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.VECTOR_BLOB_FILE_EXTENSION;
import static org.opensearch.knn.index.remote.KNNRemoteConstants.VECTOR_PATH;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.TEST_BUCKET;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.TEST_CLUSTER;
import static org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests.createMockMethodContext;

public class RemoteBuildRequestTests extends OpenSearchSingleNodeTestCase {
    public static final String MOCK_BASE_PATH = "vectors/1_1_25";
    public static final String MOCK_UUID = "SIRKos4rOWlMA62PX2p75m";
    public static final String VECTORS_PATH = "_vectors";
    public static final String MOCK_FULL_PATH = "vectors/1_1_25/SIRKos4rOWlMA62PX2p75m_vectors/SIRKos4rOWlMA62PX2p75m_target_field__3l";
    public static final String MOCK_SEGMENT_STATE = "_3l";
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
            BlobStoreRepository repository = mock(BlobStoreRepository.class);
            BlobPath baseBlobPath = new BlobPath();
            baseBlobPath = baseBlobPath.add(MOCK_BASE_PATH);
            when(repository.basePath()).thenReturn(baseBlobPath);

            // Construct the file paths exactly as created in RemoteIndexBuildStrategy
            BlobPath blobPath = repository.basePath().add(MOCK_UUID + VECTORS_PATH);
            String blobName = MOCK_UUID + "_" + indexInfo.getFieldName() + "_" + MOCK_SEGMENT_STATE;

            RemoteBuildRequest request = new RemoteBuildRequest(
                mockIndexSettings,
                indexInfo,
                metadata,
                blobPath.buildAsString() + blobName,
                createMockMethodContext()
            );

            assertEquals(S3, request.getRepositoryType());
            assertEquals(TEST_BUCKET, request.getContainerName());
            assertEquals(FAISS_NAME, request.getEngine());
            assertEquals(FLOAT.getValue(), request.getVectorDataType());
            assertEquals(MOCK_FULL_PATH + VECTOR_BLOB_FILE_EXTENSION, request.getVectorPath());
            assertEquals(MOCK_FULL_PATH + DOC_ID_FILE_EXTENSION, request.getDocIdPath());
            assertEquals(TEST_CLUSTER, request.getTenantId());
            assertEquals(2, request.getDocCount());
            assertEquals(2, request.getDimension());

            String expectedJson = getMockExpectedJson();

            // Use JSON parser to compare trees because order is not guaranteed
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

    /**
     * Get a mock JSON build request
     * <p>
     * Returns:
     * <pre>{@code {
     *   "repository_type": "s3",
     *   "container_name": "test-bucket",
     *   "vector_path": "vectors/1_1_25/SIRKos4rOWlMA62PX2p75m_vectors/SIRKos4rOWlMA62PX2p75m_target_field__3l.knnvec",
     *   "doc_id_path": "vectors/1_1_25/SIRKos4rOWlMA62PX2p75m_vectors/SIRKos4rOWlMA62PX2p75m_target_field__3l.knndid",
     *   "tenant_id": "test-cluster",
     *   "dimension": 2,
     *   "doc_count": 2,
     *   "data_type": "float",
     *   "engine": "faiss",
     *   "index_parameters": {
     *     "space_type": "l2",
     *     "algorithm": "hnsw",
     *     "algorithm_parameters": {
     *       "m": 14,
     *       "ef_construction": 94,
     *       "ef_search": 89
     *     }
     *   }
     * }}</pre>
     */
    public String getMockExpectedJson() {
        return "{"
            + "\""
            + REPOSITORY_TYPE
            + "\":\""
            + S3
            + "\","
            + "\""
            + CONTAINER_NAME
            + "\":\""
            + TEST_BUCKET
            + "\","
            + "\""
            + VECTOR_PATH
            + "\":\""
            + MOCK_FULL_PATH
            + VECTOR_BLOB_FILE_EXTENSION
            + "\","
            + "\""
            + DOC_ID_PATH
            + "\":\""
            + MOCK_FULL_PATH
            + DOC_ID_FILE_EXTENSION
            + "\","
            + "\""
            + TENANT_ID
            + "\":\""
            + TEST_CLUSTER
            + "\","
            + "\""
            + DIMENSION
            + "\":2,"
            + "\""
            + DOC_COUNT
            + "\":2,"
            + "\""
            + VECTOR_DATA_TYPE_FIELD
            + "\":\""
            + FLOAT.getValue()
            + "\","
            + "\""
            + KNN_ENGINE
            + "\":\""
            + FAISS_NAME
            + "\","
            + "\""
            + INDEX_PARAMETERS
            + "\":{"
            + "\""
            + METHOD_PARAMETER_SPACE_TYPE
            + "\":\""
            + L2.getValue()
            + "\","
            + "\""
            + ALGORITHM
            + "\":\""
            + METHOD_HNSW
            + "\","
            + "\""
            + ALGORITHM_PARAMETERS
            + "\":{"
            + "\""
            + METHOD_PARAMETER_EF_CONSTRUCTION
            + "\":94,"
            + "\""
            + METHOD_PARAMETER_EF_SEARCH
            + "\":89,"
            + "\""
            + METHOD_PARAMETER_M
            + "\":14"
            + "}"
            + "}"
            + "}";
    }
}
