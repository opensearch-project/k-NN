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

package org.opensearch.knn.plugin.transport;

import com.google.common.collect.ImmutableList;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.client.Client;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.collect.ImmutableOpenMap;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.transport.TransportService;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class TrainingJobRouterTransportActionTests extends KNNTestCase {

    public void testSingleNode_withCapacity() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of(
                "node-1"
        );

        ImmutableOpenMap<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
        ClusterService clusterService = generateMockedClusterService(discoveryNodesMap);

        // Create a response to be returned with job route decision info
        List<TrainingJobRouteDecisionInfoNodeResponse> responseList = new ArrayList<>();
        nodeIds.forEach(id -> responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(
                discoveryNodesMap.get(id),
                0 // node has capacity
        )));

        TrainingJobRouteDecisionInfoResponse infoResponse = new TrainingJobRouteDecisionInfoResponse(
                ClusterName.DEFAULT,
                responseList,
                Collections.emptyList()
        );

        TransportService transportService = mock(TransportService.class);
        Client client = mock(Client.class);

        // Setup the action
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
                transportService, new ActionFilters(Collections.emptySet()), clusterService, client);

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(null, infoResponse);
        assertEquals(nodeIds.get(0), selectedNode.getId());
    }

    public void testSingleNode_withoutCapacity() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of(
                "node-1"
        );

        ImmutableOpenMap<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
        ClusterService clusterService = generateMockedClusterService(discoveryNodesMap);

        // Create a response to be returned with job route decision info
        List<TrainingJobRouteDecisionInfoNodeResponse> responseList = new ArrayList<>();
        nodeIds.forEach(id -> responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(
                discoveryNodesMap.get(id),
                1 // node has no capacity
        )));

        TrainingJobRouteDecisionInfoResponse infoResponse = new TrainingJobRouteDecisionInfoResponse(
                ClusterName.DEFAULT,
                responseList,
                Collections.emptyList()
        );

        TransportService transportService = mock(TransportService.class);
        Client client = mock(Client.class);

        // Setup the action
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
                transportService, new ActionFilters(Collections.emptySet()), clusterService, client);

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(null, infoResponse);
        assertNull(selectedNode);
    }

    public void testMultiNode_withCapacity() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of(
                "node-1",
                "node-2",
                "node-3"
        );

        ImmutableOpenMap<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
        ClusterService clusterService = generateMockedClusterService(discoveryNodesMap);

        // Create a response to be returned with job route decision info
        List<TrainingJobRouteDecisionInfoNodeResponse> responseList = new ArrayList<>();

        // First node does not have capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(0)), 1));

        // Second node has capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(1)), 0));

        // Third node has no capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(1)), 1));

        TrainingJobRouteDecisionInfoResponse infoResponse = new TrainingJobRouteDecisionInfoResponse(
                ClusterName.DEFAULT,
                responseList,
                Collections.emptyList()
        );

        TransportService transportService = mock(TransportService.class);
        Client client = mock(Client.class);

        // Setup the action
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
                transportService, new ActionFilters(Collections.emptySet()), clusterService, client);

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(null, infoResponse);
        assertEquals(nodeIds.get(1), selectedNode.getId());
    }

    public void testMultiNode_withCapacity_withPreferredAvailable() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of(
                "node-1",
                "node-2",
                "node-3"
        );

        String preferredNode = nodeIds.get(2);

        ImmutableOpenMap<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
        ClusterService clusterService = generateMockedClusterService(discoveryNodesMap);

        // Create a response to be returned with job route decision info
        List<TrainingJobRouteDecisionInfoNodeResponse> responseList = new ArrayList<>();

        // First node has capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(0)), 0));

        // Second node has capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(1)), 0));

        // Third node with capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(2)), 0));

        TrainingJobRouteDecisionInfoResponse infoResponse = new TrainingJobRouteDecisionInfoResponse(
                ClusterName.DEFAULT,
                responseList,
                Collections.emptyList()
        );

        TransportService transportService = mock(TransportService.class);
        Client client = mock(Client.class);

        // Setup the action
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
                transportService, new ActionFilters(Collections.emptySet()), clusterService, client);

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(preferredNode, infoResponse);
        assertEquals(preferredNode, selectedNode.getId());
    }

    public void testMultiNode_withCapacity_withoutPreferredAvailable() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of(
                "node-1",
                "node-2",
                "node-3"
        );

        String preferredNode = nodeIds.get(2);

        ImmutableOpenMap<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
        ClusterService clusterService = generateMockedClusterService(discoveryNodesMap);

        // Create a response to be returned with job route decision info
        List<TrainingJobRouteDecisionInfoNodeResponse> responseList = new ArrayList<>();

        // First node has capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(0)), 0));

        // Second node has capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(1)), 0));

        // Third node with no capacity (preferred node)
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(1)), 1));

        TrainingJobRouteDecisionInfoResponse infoResponse = new TrainingJobRouteDecisionInfoResponse(
                ClusterName.DEFAULT,
                responseList,
                Collections.emptyList()
        );

        TransportService transportService = mock(TransportService.class);
        Client client = mock(Client.class);

        // Setup the action
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
                transportService, new ActionFilters(Collections.emptySet()), clusterService, client);

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(preferredNode, infoResponse);
        assertNotNull(selectedNode);
        assertNotEquals(preferredNode, selectedNode.getId());
    }

    public void testMultiNode_withoutCapacity() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of(
                "node-1",
                "node-2",
                "node-3"
        );

        ImmutableOpenMap<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
        ClusterService clusterService = generateMockedClusterService(discoveryNodesMap);

        // Create a response to be returned with job route decision info
        List<TrainingJobRouteDecisionInfoNodeResponse> responseList = new ArrayList<>();

        // First node has no capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(0)), 1));

        // Second node has no capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(1)), 1));

        // Third node has no capacity
        responseList.add(new TrainingJobRouteDecisionInfoNodeResponse(discoveryNodesMap.get(nodeIds.get(1)), 1));

        TrainingJobRouteDecisionInfoResponse infoResponse = new TrainingJobRouteDecisionInfoResponse(
                ClusterName.DEFAULT,
                responseList,
                Collections.emptyList()
        );

        TransportService transportService = mock(TransportService.class);
        Client client = mock(Client.class);

        // Setup the action
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
                transportService, new ActionFilters(Collections.emptySet()), clusterService, client);

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(null, infoResponse);
        assertNull(selectedNode);
    }

    private ImmutableOpenMap<String, DiscoveryNode> generateDiscoveryNodes(List<String> dataNodeIds) {
        ImmutableOpenMap.Builder<String, DiscoveryNode> builder = ImmutableOpenMap.builder();

        for (String nodeId : dataNodeIds) {
            DiscoveryNode discoveryNode = mock(DiscoveryNode.class);
            when(discoveryNode.getId()).thenReturn(nodeId);
            builder.put(nodeId, discoveryNode);
        }

        return builder.build();
    }

    private ClusterService generateMockedClusterService(ImmutableOpenMap<String, DiscoveryNode> discoveryNodeMap) {
        DiscoveryNodes discoveryNodes = mock(DiscoveryNodes.class);
        when(discoveryNodes.getDataNodes()).thenReturn(discoveryNodeMap);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.nodes()).thenReturn(discoveryNodes);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        return clusterService;
    }
}
