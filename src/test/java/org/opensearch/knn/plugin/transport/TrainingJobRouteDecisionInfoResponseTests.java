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
import com.google.common.collect.Maps;
import com.google.common.net.InetAddresses;
import org.opensearch.Version;
import org.opensearch.action.FailedNodeException;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.transport.TransportAddress;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.net.InetAddress;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.NODES_KEY;
import static org.opensearch.knn.common.KNNConstants.TRAINING_JOB_COUNT_FIELD_NAME;

public class TrainingJobRouteDecisionInfoResponseTests extends KNNTestCase {

    public void testStreams() throws IOException {

        // Initialize nodes and data
        InetAddress inetAddress1 = InetAddresses.fromInteger(randomInt());
        String node1Id = "node-1";
        DiscoveryNode discoveryNode1 = new DiscoveryNode(node1Id, new TransportAddress(inetAddress1, 9200), Version.CURRENT);
        Integer trainingJobCount1 = 1;
        TrainingJobRouteDecisionInfoNodeResponse nodeResponse1 = new TrainingJobRouteDecisionInfoNodeResponse(
                discoveryNode1,
                trainingJobCount1
        );

        InetAddress inetAddress2 = InetAddresses.fromInteger(randomInt());
        String node2Id = "node-2";
        DiscoveryNode discoveryNode2 = new DiscoveryNode(node2Id, new TransportAddress(inetAddress2, 9200), Version.CURRENT);
        Integer trainingJobCount2 = 2;
        TrainingJobRouteDecisionInfoNodeResponse nodeResponse2 = new TrainingJobRouteDecisionInfoNodeResponse(
                discoveryNode2,
                trainingJobCount2
        );

        InetAddress inetAddress3 = InetAddresses.fromInteger(randomInt());
        String node3Id = "node-3";
        DiscoveryNode discoveryNode3 = new DiscoveryNode(node3Id, new TransportAddress(inetAddress3, 9200), Version.CURRENT);
        Integer trainingJobCount3 = 3;
        TrainingJobRouteDecisionInfoNodeResponse nodeResponse3 = new TrainingJobRouteDecisionInfoNodeResponse(
                discoveryNode3,
                trainingJobCount3
        );

        List<TrainingJobRouteDecisionInfoNodeResponse> nodeResponses = ImmutableList.of(nodeResponse1, nodeResponse2, nodeResponse3);

        List<FailedNodeException> failedNodeExceptions = Collections.emptyList();

        // Setup output
        BytesStreamOutput streamOutput = new BytesStreamOutput();

        TrainingJobRouteDecisionInfoResponse original = new TrainingJobRouteDecisionInfoResponse(
                ClusterName.DEFAULT,
                nodeResponses,
                failedNodeExceptions
        );

        original.writeTo(streamOutput);

        // Read back streamed out into streamed in
        TrainingJobRouteDecisionInfoResponse copy = new TrainingJobRouteDecisionInfoResponse(streamOutput.bytes().streamInput());

        Map<String, TrainingJobRouteDecisionInfoNodeResponse> originalNodeResponseMap = original.getNodesMap();
        Map<String, TrainingJobRouteDecisionInfoNodeResponse> copyNodeResponseMap = copy.getNodesMap();
        assertEquals(originalNodeResponseMap.keySet(), copyNodeResponseMap.keySet());
        assertTrue(originalNodeResponseMap.containsKey(node2Id));
        assertEquals(originalNodeResponseMap.get(node2Id).getTrainingJobCount(), copyNodeResponseMap.get(node2Id).getTrainingJobCount());
    }

    public void testToXContent() throws IOException {

        // Set up the data
        String id1 = "id_1";
        DiscoveryNode discoveryNode1 = mock(DiscoveryNode.class);
        when(discoveryNode1.getId()).thenReturn(id1);
        Integer trainingJobCount1 = 1;
        TrainingJobRouteDecisionInfoNodeResponse nodeResponse1 = new TrainingJobRouteDecisionInfoNodeResponse(
                discoveryNode1,
                trainingJobCount1
        );

        String id2 = "id_2";
        DiscoveryNode discoveryNode2 = mock(DiscoveryNode.class);
        when(discoveryNode2.getId()).thenReturn(id2);
        Integer trainingJobCount2 = 2;
        TrainingJobRouteDecisionInfoNodeResponse nodeResponse2 = new TrainingJobRouteDecisionInfoNodeResponse(
                discoveryNode2,
                trainingJobCount2
        );

        String id3 = "id_3";
        DiscoveryNode discoveryNode3 = mock(DiscoveryNode.class);
        when(discoveryNode3.getId()).thenReturn(id3);
        Integer trainingJobCount3 = 3;
        TrainingJobRouteDecisionInfoNodeResponse nodeResponse3 = new TrainingJobRouteDecisionInfoNodeResponse(
                discoveryNode3,
                trainingJobCount3
        );

        // We expect this:
        // {
        // "nodes": {
        // "id_1": {
        // "training_job_count": 1
        // },
        // "id_2": {
        // "training_job_count": 2
        // },
        // "id_3": {
        // "training_job_count": 3
        // },
        // }
        // }
        XContentBuilder expectedXContentBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(NODES_KEY)
                .startObject(id1)
                .field(TRAINING_JOB_COUNT_FIELD_NAME, trainingJobCount1)
                .endObject()
                .startObject(id2)
                .field(TRAINING_JOB_COUNT_FIELD_NAME, trainingJobCount2)
                .endObject()
                .startObject(id3)
                .field(TRAINING_JOB_COUNT_FIELD_NAME, trainingJobCount3)
                .endObject()
                .endObject()
                .endObject();
        Map<String, Object> expected = xContentBuilderToMap(expectedXContentBuilder);

        // Configure response
        List<TrainingJobRouteDecisionInfoNodeResponse> nodeResponses = ImmutableList.of(nodeResponse1, nodeResponse2, nodeResponse3);

        List<FailedNodeException> failedNodeExceptions = Collections.emptyList();

        TrainingJobRouteDecisionInfoResponse response = new TrainingJobRouteDecisionInfoResponse(
                ClusterName.DEFAULT,
                nodeResponses,
                failedNodeExceptions
        );

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder = response.toXContent(builder, ToXContent.EMPTY_PARAMS).endObject();
        Map<String, Object> actual = xContentBuilderToMap(builder);

        // Check responses are equal
        assertTrue(Maps.difference(expected, actual).areEqual());
    }
}