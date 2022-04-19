/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Map;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.index.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_EF_SEARCH;

public class IndexUtilTests extends KNNTestCase {
    public void testGetLoadParameters() {
        // Test faiss to ensure that space type gets set properly
        SpaceType spaceType1 = SpaceType.COSINESIMIL;
        KNNEngine knnEngine1 = KNNEngine.FAISS;
        String indexName = "my-test-index";

        Map<String, Object> loadParameters = getParametersAtLoading(spaceType1, knnEngine1, indexName);
        assertEquals(1, loadParameters.size());
        assertEquals(spaceType1.getValue(), loadParameters.get(SPACE_TYPE));

        // Test nmslib to ensure both space type and ef search are properly set
        SpaceType spaceType2 = SpaceType.L1;
        KNNEngine knnEngine2 = KNNEngine.NMSLIB;
        int efSearchValue = 413;

        // We use the constant for the setting here as opposed to the identifier of efSearch in nmslib jni
        Map<String, Object> indexSettings = ImmutableMap.of(KNN_ALGO_PARAM_EF_SEARCH, efSearchValue);

        // Because ef search comes from an index setting, we need to mock the long line of calls to get those
        // index settings
        Settings settings = Settings.builder().loadFromMap(indexSettings).build();
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        when(indexMetadata.getSettings()).thenReturn(settings);
        Metadata metadata = mock(Metadata.class);
        when(metadata.index(anyString())).thenReturn(indexMetadata);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.getMetadata()).thenReturn(metadata);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);
        KNNSettings.state().setClusterService(clusterService);

        loadParameters = getParametersAtLoading(spaceType2, knnEngine2, indexName);
        assertEquals(2, loadParameters.size());
        assertEquals(spaceType2.getValue(), loadParameters.get(SPACE_TYPE));
        assertEquals(efSearchValue, loadParameters.get(HNSW_ALGO_EF_SEARCH));
    }
}
