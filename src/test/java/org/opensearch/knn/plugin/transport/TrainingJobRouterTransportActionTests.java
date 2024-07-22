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
import org.apache.lucene.search.TotalHits;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.client.Client;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.search.SearchHit;
import org.opensearch.search.SearchHits;
import org.opensearch.transport.TransportService;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;

public class TrainingJobRouterTransportActionTests extends KNNTestCase {

    public void testSingleNode_withCapacity() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of("node-1");

        Map<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
        ClusterService clusterService = generateMockedClusterService(discoveryNodesMap);

        // Create a response to be returned with job route decision info
        List<TrainingJobRouteDecisionInfoNodeResponse> responseList = new ArrayList<>();
        nodeIds.forEach(
            id -> responseList.add(
                new TrainingJobRouteDecisionInfoNodeResponse(
                    discoveryNodesMap.get(id),
                    0 // node has capacity
                )
            )
        );

        TrainingJobRouteDecisionInfoResponse infoResponse = new TrainingJobRouteDecisionInfoResponse(
            ClusterName.DEFAULT,
            responseList,
            Collections.emptyList()
        );

        TransportService transportService = mock(TransportService.class);
        Client client = mock(Client.class);

        // Setup the action
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(null, infoResponse);
        assertEquals(nodeIds.get(0), selectedNode.getId());
    }

    public void testSingleNode_withoutCapacity() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of("node-1");

        Map<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
        ClusterService clusterService = generateMockedClusterService(discoveryNodesMap);

        // Create a response to be returned with job route decision info
        List<TrainingJobRouteDecisionInfoNodeResponse> responseList = new ArrayList<>();
        nodeIds.forEach(
            id -> responseList.add(
                new TrainingJobRouteDecisionInfoNodeResponse(
                    discoveryNodesMap.get(id),
                    1 // node has no capacity
                )
            )
        );

        TrainingJobRouteDecisionInfoResponse infoResponse = new TrainingJobRouteDecisionInfoResponse(
            ClusterName.DEFAULT,
            responseList,
            Collections.emptyList()
        );

        TransportService transportService = mock(TransportService.class);
        Client client = mock(Client.class);

        // Setup the action
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(null, infoResponse);
        assertNull(selectedNode);
    }

    public void testMultiNode_withCapacity() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of("node-1", "node-2", "node-3");

        Map<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
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
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(null, infoResponse);
        assertEquals(nodeIds.get(1), selectedNode.getId());
    }

    public void testMultiNode_withCapacity_withPreferredAvailable() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of("node-1", "node-2", "node-3");

        String preferredNode = nodeIds.get(2);

        Map<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
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
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(preferredNode, infoResponse);
        assertEquals(preferredNode, selectedNode.getId());
    }

    public void testMultiNode_withCapacity_withoutPreferredAvailable() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of("node-1", "node-2", "node-3");

        String preferredNode = nodeIds.get(2);

        Map<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
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
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(preferredNode, infoResponse);
        assertNotNull(selectedNode);
        assertNotEquals(preferredNode, selectedNode.getId());
    }

    public void testMultiNode_withoutCapacity() {
        // Mock datanodes in the cluster through mocking the cluster service
        List<String> nodeIds = ImmutableList.of("node-1", "node-2", "node-3");

        Map<String, DiscoveryNode> discoveryNodesMap = generateDiscoveryNodes(nodeIds);
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
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        // Select the node
        DiscoveryNode selectedNode = transportAction.selectNode(null, infoResponse);
        assertNull(selectedNode);
    }

    @SuppressWarnings("unchecked")
    public void testTrainingIndexSize() {

        String trainingIndexName = "training-index";
        int dimension = 133;
        int vectorCount = 1000000;
        int expectedSize = dimension * vectorCount * Float.BYTES / BYTES_PER_KILOBYTES + 1; // 519,531.25 KB ~= 520 MB

        // Setup the request
        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            null,
            KNNMethodContext.getDefault(),
            dimension,
            trainingIndexName,
            "training-field",
            null,
            "description",
            VectorDataType.DEFAULT
        );

        // Mock client to return the right number of docs
        TotalHits totalHits = new TotalHits(vectorCount, TotalHits.Relation.EQUAL_TO);
        SearchHits searchHits = new SearchHits(new SearchHit[2], totalHits, 1.0f);
        SearchResponse searchResponse = mock(SearchResponse.class);
        when(searchResponse.getHits()).thenReturn(searchHits);
        Client client = mock(Client.class);
        doAnswer(invocationOnMock -> {
            ((ActionListener<SearchResponse>) invocationOnMock.getArguments()[1]).onResponse(searchResponse);
            return null;
        }).when(client).search(any(), any());

        // Setup the action
        ClusterService clusterService = mock(ClusterService.class);
        TransportService transportService = mock(TransportService.class);
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        ActionListener<Integer> listener = ActionListener.wrap(
            size -> assertEquals(expectedSize, size.intValue()),
            e -> fail(e.getMessage())
        );

        transportAction.getTrainingIndexSizeInKB(trainingModelRequest, listener);
    }

    public void testTrainIndexSize_whenDataTypeIsBinary() {
        String trainingIndexName = "training-index";
        int dimension = 8;
        int vectorCount = 1000000;
        int expectedSize = Byte.BYTES * (dimension / 8) * vectorCount / BYTES_PER_KILOBYTES + 1; // 977 KB

        // Setup the request
        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            null,
            KNNMethodContext.getDefault(),
            dimension,
            trainingIndexName,
            "training-field",
            null,
            "description",
            VectorDataType.BINARY
        );

        // Mock client to return the right number of docs
        TotalHits totalHits = new TotalHits(vectorCount, TotalHits.Relation.EQUAL_TO);
        SearchHits searchHits = new SearchHits(new SearchHit[2], totalHits, 1.0f);
        SearchResponse searchResponse = mock(SearchResponse.class);
        when(searchResponse.getHits()).thenReturn(searchHits);
        Client client = mock(Client.class);

        doAnswer(invocationOnMock -> {
            ((ActionListener<SearchResponse>) invocationOnMock.getArguments()[1]).onResponse(searchResponse);
            return null;
        }).when(client).search(any(), any());

        // Setup the action
        ClusterService clusterService = mock(ClusterService.class);
        TransportService transportService = mock(TransportService.class);
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        ActionListener<Integer> listener = ActionListener.wrap(
            size -> assertEquals(expectedSize, size.intValue()),
            e -> fail(e.getMessage())
        );

        transportAction.getTrainingIndexSizeInKB(trainingModelRequest, listener);
    }

    public void testTrainIndexSize_whenDataTypeIsByte() {
        String trainingIndexName = "training-index";
        int dimension = 8;
        int vectorCount = 1000000;
        int expectedSize = Byte.BYTES * dimension * vectorCount / BYTES_PER_KILOBYTES + 1; // 7813 KB

        // Setup the request
        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            null,
            KNNMethodContext.getDefault(),
            dimension,
            trainingIndexName,
            "training-field",
            null,
            "description",
            VectorDataType.BYTE
        );

        // Mock client to return the right number of docs
        TotalHits totalHits = new TotalHits(vectorCount, TotalHits.Relation.EQUAL_TO);
        SearchHits searchHits = new SearchHits(new SearchHit[2], totalHits, 1.0f);
        SearchResponse searchResponse = mock(SearchResponse.class);
        when(searchResponse.getHits()).thenReturn(searchHits);
        Client client = mock(Client.class);

        doAnswer(invocationOnMock -> {
            ((ActionListener<SearchResponse>) invocationOnMock.getArguments()[1]).onResponse(searchResponse);
            return null;
        }).when(client).search(any(), any());

        // Setup the action
        ClusterService clusterService = mock(ClusterService.class);
        TransportService transportService = mock(TransportService.class);
        TrainingJobRouterTransportAction transportAction = new TrainingJobRouterTransportAction(
            transportService,
            new ActionFilters(Collections.emptySet()),
            clusterService,
            client
        );

        ActionListener<Integer> listener = ActionListener.wrap(
            size -> assertEquals(expectedSize, size.intValue()),
            e -> fail(e.getMessage())
        );

        transportAction.getTrainingIndexSizeInKB(trainingModelRequest, listener);
    }

    private Map<String, DiscoveryNode> generateDiscoveryNodes(List<String> dataNodeIds) {
        Map<String, DiscoveryNode> nodes = new HashMap<>();

        for (String nodeId : dataNodeIds) {
            DiscoveryNode discoveryNode = mock(DiscoveryNode.class);
            when(discoveryNode.getId()).thenReturn(nodeId);
            nodes.put(nodeId, discoveryNode);
        }

        return nodes;
    }

    private ClusterService generateMockedClusterService(Map<String, DiscoveryNode> discoveryNodeMap) {
        DiscoveryNodes discoveryNodes = mock(DiscoveryNodes.class);
        when(discoveryNodes.getDataNodes()).thenReturn(discoveryNodeMap);
        ClusterState clusterState = mock(ClusterState.class);
        when(clusterState.nodes()).thenReturn(discoveryNodes);
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(clusterState);

        return clusterService;
    }
}
