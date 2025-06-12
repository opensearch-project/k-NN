/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.SetOnce;
import org.opensearch.common.blobstore.BlobContainer;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.common.blobstore.BlobStore;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.core.index.Index;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue;
import org.opensearch.remoteindexbuild.model.RemoteBuildRequest;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.Version.CURRENT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_BUILD_SIZE_MAX_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPOSITORY_SETTING;
import static org.opensearch.knn.index.SpaceType.INNER_PRODUCT;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_FLUSH_TIME;
import static org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_MERGE_TIME;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.DOC_ID_FILE_EXTENSION;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_ENCODER;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.S3;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.VECTOR_BLOB_FILE_EXTENSION;

public class RemoteIndexBuildStrategyTests extends RemoteIndexBuildTests {
    private static final String TEST_INDEX = "test-index";
    public static final String MOCK_BASE_PATH = "vectors/1_1_25";
    public static final String MOCK_UUID = "SIRKos4rOWlMA62PX2p75m";
    public static final String VECTORS_PATH = "_vectors";
    public static final String MOCK_FULL_PATH = "vectors/1_1_25/SIRKos4rOWlMA62PX2p75m_vectors/SIRKos4rOWlMA62PX2p75m_target_field__3l";
    public static final String MOCK_SEGMENT_STATE = "_3l";

    /**
     * Test that we fallback to the fallback NativeIndexBuildStrategy when an exception is thrown
     */
    public void testRemoteIndexBuildStrategyFallback() throws IOException {
        RepositoriesService repositoriesService = mock(RepositoriesService.class);
        when(repositoriesService.repository(any())).thenThrow(new RepositoryMissingException("Fallback"));

        final SetOnce<Boolean> fallback = new SetOnce<>();
        RemoteIndexBuildStrategy objectUnderTest = new RemoteIndexBuildStrategy(
            () -> repositoriesService,
            new TestIndexBuildStrategy(fallback),
            mock(IndexSettings.class),
            null
        );
        objectUnderTest.buildAndWriteIndex(buildIndexParams);
        assertTrue(fallback.get());
        for (KNNRemoteIndexBuildValue value : KNNRemoteIndexBuildValue.values()) {
            if (value == REMOTE_INDEX_BUILD_FLUSH_TIME && buildIndexParams.isFlush()) {
                assertTrue(value.getValue() >= 0L);
            } else if (value == REMOTE_INDEX_BUILD_MERGE_TIME && !buildIndexParams.isFlush()) {
                assertTrue(value.getValue() >= 0L);
            } else if (value == KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT) {
                assertEquals(1L, (long) value.getValue());
            } else {
                assertEquals(0L, (long) value.getValue());
            }
        }
    }

    public void testShouldBuildIndexRemotely() {
        IndexSettings indexSettings;
        ClusterSettings clusterSettings;
        Index index = mock(Index.class);
        when(index.getName()).thenReturn(TEST_INDEX);
        // Check index settings null
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(null, 0));

        // Check index setting disabled
        Settings settings = Settings.builder()
            .put(settings(CURRENT).build())
            .put(KNN_INDEX, true)
            .put(KNN_INDEX_REMOTE_VECTOR_BUILD, false)
            .build();
        IndexMetadata metadata = IndexMetadata.builder("test-index")
            .settings(settings)
            .numberOfShards(1)
            .numberOfReplicas(0)
            .version(7)
            .mappingVersion(0)
            .settingsVersion(0)
            .aliasesVersion(0)
            .creationDate(0)
            .build();
        indexSettings = new IndexSettings(
            metadata,
            Settings.EMPTY,
            new IndexScopedSettings(Settings.EMPTY, new HashSet<>(IndexScopedSettings.BUILT_IN_INDEX_SETTINGS))
        );
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, 0));

        // Check repo not configured
        clusterSettings = mock(ClusterSettings.class);
        when(clusterSettings.get(KNN_REMOTE_VECTOR_REPOSITORY_SETTING)).thenReturn("");
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, 0));

        // Check size threshold
        int BYTE_SIZE = randomIntBetween(50, 1000);
        settings = Settings.builder()
            .put(settings(CURRENT).build())
            .put(KNN_INDEX, true)
            .put(KNN_INDEX_REMOTE_VECTOR_BUILD, false)
            .put(KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN, BYTE_SIZE)
            .build();
        metadata = IndexMetadata.builder("test-index")
            .settings(settings)
            .numberOfShards(1)
            .numberOfReplicas(0)
            .version(7)
            .mappingVersion(0)
            .settingsVersion(0)
            .aliasesVersion(0)
            .creationDate(0)
            .build();
        indexSettings.updateIndexMetadata(metadata);
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, randomInt(BYTE_SIZE - 1)));

        // Check happy path
        settings = Settings.builder()
            .put(settings(CURRENT).build())
            .put(KNN_INDEX, true)
            .put(KNN_INDEX_REMOTE_VECTOR_BUILD, true)
            .put(KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN, new ByteSizeValue(BYTE_SIZE))
            .build();
        metadata = IndexMetadata.builder("test-index")
            .settings(settings)
            .numberOfShards(1)
            .numberOfReplicas(0)
            .version(7)
            .mappingVersion(0)
            .settingsVersion(0)
            .aliasesVersion(0)
            .creationDate(0)
            .build();
        Set<Setting<?>> indexScopedSettings = new HashSet<>(IndexScopedSettings.BUILT_IN_INDEX_SETTINGS);
        indexScopedSettings.add(KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN_SETTING);
        indexSettings = new IndexSettings(metadata, settings, new IndexScopedSettings(settings, indexScopedSettings));
        clusterSettings = mock(ClusterSettings.class);
        when(clusterSettings.get(KNN_REMOTE_VECTOR_REPOSITORY_SETTING)).thenReturn("test-vector-repo");
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
        when(clusterSettings.get(KNN_REMOTE_VECTOR_BUILD_SIZE_MAX_SETTING)).thenReturn(new ByteSizeValue(BYTE_SIZE * 3L));
        assertTrue(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, randomIntBetween(BYTE_SIZE, BYTE_SIZE * 2)));
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, randomIntBetween(BYTE_SIZE * 3 + 1, BYTE_SIZE * 4)));

        // Check index setting not set resolves to enabled
        settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        metadata = IndexMetadata.builder("test-index")
            .settings(settings)
            .numberOfShards(1)
            .numberOfReplicas(0)
            .version(7)
            .mappingVersion(0)
            .settingsVersion(0)
            .aliasesVersion(0)
            .creationDate(0)
            .build();
        indexSettings.updateIndexMetadata(metadata);
        assertTrue(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, randomIntBetween(BYTE_SIZE, BYTE_SIZE * 2)));
    }

    public void testFilePathConstruction() {
        BlobStoreRepository repository = mock(BlobStoreRepository.class);
        BlobStore mockBlobStore = mock(BlobStore.class);
        when(repository.blobStore()).thenReturn(mockBlobStore);

        BlobPath baseBlobPath = new BlobPath().add(MOCK_BASE_PATH);
        when(repository.basePath()).thenReturn(baseBlobPath);

        IndexSettings indexSettings = createTestIndexSettings();
        BlobPath blobPath = repository.basePath().add(indexSettings.getUUID() + VECTORS_PATH);

        BlobContainer blobContainer = mock(BlobContainer.class);
        when(mockBlobStore.blobContainer(blobPath)).thenReturn(blobContainer);
        when(blobContainer.path()).thenReturn(blobPath);

        String blobName = MOCK_UUID + "_" + buildIndexParams.getFieldName() + "_" + MOCK_SEGMENT_STATE;

        // example: VectorRepositoryAccessor vectorAccessor = new DefaultVectorRepositoryAccessor(blobContainer);
        // vectorAccessor.writeToRepository(blobName...)
        String accessorPath = blobContainer.path().buildAsString();

        String requestPath = blobPath.buildAsString() + blobName;
        assertEquals(accessorPath + blobName, requestPath);
    }

    public void testBuildRequest() throws IOException {
        RemoteBuildRequest request = RemoteIndexBuildStrategy.buildRemoteBuildRequest(
            createTestIndexSettings(),
            buildIndexParams,
            createTestRepositoryMetadata(),
            MOCK_FULL_PATH,
            getMockParameterMap()
        );
        assertEquals(S3, request.getRepositoryType());
        assertEquals(TEST_BUCKET, request.getContainerName());
        assertEquals(KNNConstants.FAISS_NAME, request.getEngine());
        assertEquals(VectorDataType.FLOAT.getValue(), request.getVectorDataType());
        assertEquals(MOCK_FULL_PATH + VECTOR_BLOB_FILE_EXTENSION, request.getVectorPath());
        assertEquals(MOCK_FULL_PATH + DOC_ID_FILE_EXTENSION, request.getDocIdPath());
        assertEquals(TEST_CLUSTER, request.getTenantId());
        assertEquals(3, request.getDocCount());
        assertEquals(2, request.getDimension());
    }

    public Map<String, Object> getMockParameterMap() {
        Map<String, Object> encoderSq = Map.of(ENCODER_SQ, Map.of());
        Map<String, Object> encoderMap = Map.of(METHOD_ENCODER_PARAMETER, encoderSq);
        Map<String, Object> innerParams = Map.of(
            METHOD_PARAMETER_EF_SEARCH,
            24,
            METHOD_PARAMETER_EF_CONSTRUCTION,
            28,
            METHOD_PARAMETER_M,
            12,
            METHOD_PARAMETER_ENCODER,
            encoderMap
        );
        return Map.of(
            INDEX_DESCRIPTION_PARAMETER,
            "HNSW12,Flat",
            SPACE_TYPE,
            INNER_PRODUCT.getValue(),
            NAME,
            METHOD_HNSW,
            VECTOR_DATA_TYPE_FIELD,
            VectorDataType.BYTE.getValue(),
            PARAMETERS,
            innerParams
        );
    }
}
