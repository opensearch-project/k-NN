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

import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.nodes.BaseNodesResponse;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.NODES_KEY;

/**
 * Aggregated response for training job route decision info
 */
public class TrainingJobRouteDecisionInfoResponse extends BaseNodesResponse<TrainingJobRouteDecisionInfoNodeResponse>
    implements
        ToXContentObject {

    /**
     * Constructor
     *
     * @param in StreamInput
     * @throws IOException thrown when unable to read from stream
     */
    public TrainingJobRouteDecisionInfoResponse(StreamInput in) throws IOException {
        super(new ClusterName(in), in.readList(TrainingJobRouteDecisionInfoNodeResponse::new), in.readList(FailedNodeException::new));
    }

    /**
     * Constructor
     *
     * @param clusterName name of cluster
     * @param nodes List of KNNStatsNodeResponses
     * @param failures List of failures from nodes
     */
    public TrainingJobRouteDecisionInfoResponse(
        ClusterName clusterName,
        List<TrainingJobRouteDecisionInfoNodeResponse> nodes,
        List<FailedNodeException> failures
    ) {
        super(clusterName, nodes, failures);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
    }

    @Override
    public void writeNodesTo(StreamOutput out, List<TrainingJobRouteDecisionInfoNodeResponse> nodes) throws IOException {
        out.writeList(nodes);
    }

    @Override
    public List<TrainingJobRouteDecisionInfoNodeResponse> readNodesFrom(StreamInput in) throws IOException {
        return in.readList(TrainingJobRouteDecisionInfoNodeResponse::new);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        // Add node responses to response
        String nodeId;
        DiscoveryNode node;
        builder.startObject(NODES_KEY);
        for (TrainingJobRouteDecisionInfoNodeResponse jobRouteDecInfo : getNodes()) {
            node = jobRouteDecInfo.getNode();
            nodeId = node.getId();
            builder.startObject(nodeId);
            jobRouteDecInfo.toXContent(builder, params);
            builder.endObject();
        }
        builder.endObject();
        return builder;
    }
}
