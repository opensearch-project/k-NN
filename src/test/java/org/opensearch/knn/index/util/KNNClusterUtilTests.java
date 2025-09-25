/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;

public class KNNClusterUtilTests extends KNNTestCase {

    public void testSingleNodeCluster() {
        ClusterService clusterService = mockClusterService(Version.V_2_4_0);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService, mock(IndexNameExpressionResolver.class));

        final Version minVersion = knnClusterUtil.getClusterMinVersion();

        assertTrue(Version.V_2_4_0.equals(minVersion));
    }

    public void testMultipleNodesCluster() {
        ClusterService clusterService = mockClusterService(Version.V_2_3_0);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService, mock(IndexNameExpressionResolver.class));

        final Version minVersion = knnClusterUtil.getClusterMinVersion();

        assertTrue(Version.V_2_3_0.equals(minVersion));
    }

    public void testWhenErrorOnClusterStateDiscover() {
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(
            new ClusterSettings(Settings.EMPTY, ClusterSettings.BUILT_IN_CLUSTER_SETTINGS)
        );
        when(clusterService.state()).thenThrow(new RuntimeException("Cluster state is not ready"));

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService, mock(IndexNameExpressionResolver.class));

        final Version minVersion = knnClusterUtil.getClusterMinVersion();

        assertTrue(Version.CURRENT.equals(minVersion));
    }
}
