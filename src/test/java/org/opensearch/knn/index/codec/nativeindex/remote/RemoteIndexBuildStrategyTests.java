/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.opensearch.common.SetOnce;
import org.opensearch.common.blobstore.BlobContainer;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.common.blobstore.BlobStore;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.core.index.Index;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.remote.RemoteIndexHTTPClientTests;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.RepositoryMissingException;
import org.opensearch.repositories.blobstore.BlobStoreRepository;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;
import static org.opensearch.knn.index.remote.RemoteBuildRequestTests.MOCK_BASE_PATH;
import static org.opensearch.knn.index.remote.RemoteBuildRequestTests.MOCK_SEGMENT_STATE;
import static org.opensearch.knn.index.remote.RemoteBuildRequestTests.MOCK_UUID;
import static org.opensearch.knn.index.remote.RemoteBuildRequestTests.VECTORS_PATH;

public class RemoteIndexBuildStrategyTests extends RemoteIndexBuildTests {
    private static final String TEST_INDEX = "test-index";

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
    }

    public void testShouldBuildIndexRemotely() {
        IndexSettings indexSettings;
        ClusterSettings clusterSettings;
        Index index = mock(Index.class);
        when(index.getName()).thenReturn(TEST_INDEX);
        // Check index settings null
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(null, 0));

        // Check index setting disabled
        indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING)).thenReturn(false);
        when(indexSettings.getIndex()).thenReturn(index);
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, 0));

        // Check repo not configured
        indexSettings = mock(IndexSettings.class);
        when(indexSettings.getIndex()).thenReturn(index);
        when(indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING)).thenReturn(true);
        clusterSettings = mock(ClusterSettings.class);
        when(clusterSettings.get(KNN_REMOTE_VECTOR_REPO_SETTING)).thenReturn("");
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, 0));

        // Check size threshold
        int BYTE_SIZE = randomIntBetween(50, 1000);
        when(indexSettings.getValue(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_SETTING)).thenReturn(new ByteSizeValue(BYTE_SIZE));
        assertFalse(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, randomInt(BYTE_SIZE - 1)));

        // Check happy path
        clusterSettings = mock(ClusterSettings.class);
        when(clusterSettings.get(KNN_REMOTE_VECTOR_REPO_SETTING)).thenReturn("test-vector-repo");
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
        assertTrue(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, randomIntBetween(BYTE_SIZE - 1, BYTE_SIZE * 2)));
    }

    public void testFilePathConstruction() {
        BlobStoreRepository repository = mock(BlobStoreRepository.class);
        BlobStore mockBlobStore = mock(BlobStore.class);
        when(repository.blobStore()).thenReturn(mockBlobStore);

        BlobPath baseBlobPath = new BlobPath().add(MOCK_BASE_PATH);
        when(repository.basePath()).thenReturn(baseBlobPath);

        IndexSettings indexSettings = RemoteIndexHTTPClientTests.createTestIndexSettings();
        BlobPath blobPath = repository.basePath().add(indexSettings.getUUID() + VECTORS_PATH);

        BlobContainer blobContainer = mock(BlobContainer.class);
        when(mockBlobStore.blobContainer(blobPath)).thenReturn(blobContainer);
        when(blobContainer.path()).thenReturn(blobPath);

        BuildIndexParams indexInfo = RemoteIndexHTTPClientTests.createTestBuildIndexParams();
        String blobName = MOCK_UUID + "_" + indexInfo.getFieldName() + "_" + MOCK_SEGMENT_STATE;

        // example: VectorRepositoryAccessor vectorAccessor = new DefaultVectorRepositoryAccessor(blobContainer);
        // vectorAccessor.writeToRepository(blobName...)
        String accessorPath = blobContainer.path().buildAsString();

        String requestPath = blobPath.buildAsString() + blobName;
        assertEquals(accessorPath + blobName, requestPath);
    }
}
