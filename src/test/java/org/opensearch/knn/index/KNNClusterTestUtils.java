/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.Version;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.collect.ImmutableOpenMap;

import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.test.OpenSearchTestCase.randomAlphaOfLength;

/**
 * Collection of util methods required for testing and related to OpenSearch cluster setup and functionality
 */
public class KNNClusterTestUtils {

    /**
     * Create new mock for ClusterService
     * @param versions list of versions for cluster nodes
     * @return
     */
    public static ClusterService mockClusterService(final List<Version> versions) {
        ClusterService clusterService = mock(ClusterService.class);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterService.state()).thenReturn(clusterState);
        DiscoveryNodes discoveryNodes = mock(DiscoveryNodes.class);
        when(clusterState.getNodes()).thenReturn(discoveryNodes);
        ImmutableOpenMap.Builder<String, DiscoveryNode> builder = ImmutableOpenMap.builder();
        for (Version version : versions) {
            DiscoveryNode clusterNode = mock(DiscoveryNode.class);
            when(clusterNode.getVersion()).thenReturn(version);
            builder.put(randomAlphaOfLength(10), clusterNode);
        }
        ImmutableOpenMap<String, DiscoveryNode> mapOfNodes = builder.build();
        when(discoveryNodes.getNodes()).thenReturn(mapOfNodes);

        return clusterService;
    }
}
