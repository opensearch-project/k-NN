/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.remote;

import org.opensearch.common.SetOnce;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.core.index.Index;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.RepositoryMissingException;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_SETTING;
import static org.opensearch.knn.index.KNNSettings.KNN_REMOTE_VECTOR_REPO_SETTING;

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
            mock(IndexSettings.class)
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
        assertTrue(RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, randomIntBetween(BYTE_SIZE, BYTE_SIZE * 2)));
    }
}
