/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.Version;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;

public class KNNClusterContextTests extends KNNTestCase {

    public void testSingleNodeCluster() {
        ClusterService clusterService = mockClusterService(Version.V_2_4_0);

        final KNNClusterContext knnClusterContext = KNNClusterContext.instance();
        knnClusterContext.initialize(clusterService);

        final Version minVersion = knnClusterContext.getClusterMinVersion();

        assertTrue(Version.V_2_4_0.equals(minVersion));
    }

    public void testMultipleNodesCluster() {
        ClusterService clusterService = mockClusterService(Version.V_2_3_0);

        final KNNClusterContext knnClusterContext = KNNClusterContext.instance();
        knnClusterContext.initialize(clusterService);

        final Version minVersion = knnClusterContext.getClusterMinVersion();

        assertTrue(Version.V_2_3_0.equals(minVersion));
    }

    public void testWhenErrorOnClusterStateDiscover() {
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenThrow(new RuntimeException("Cluster state is not ready"));

        final KNNClusterContext knnClusterContext = KNNClusterContext.instance();
        knnClusterContext.initialize(clusterService);

        final Version minVersion = knnClusterContext.getClusterMinVersion();

        assertTrue(Version.CURRENT.equals(minVersion));
    }
}
