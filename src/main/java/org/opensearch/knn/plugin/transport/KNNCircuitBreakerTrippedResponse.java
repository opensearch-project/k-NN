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

import lombok.Getter;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.nodes.BaseNodesResponse;
import org.opensearch.cluster.ClusterName;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;
import java.util.List;

/**
 * Response indicating if circuit breaker has been tripped. Circuit breaker is said to be tripped if it is tripped
 * on any nodes.
 */
@Getter
public class KNNCircuitBreakerTrippedResponse extends BaseNodesResponse<KNNCircuitBreakerTrippedNodeResponse> {

    private final boolean isTripped;

    /**
     * Constructor.
     *
     * @param clusterName cluster's name
     * @param nodes list of responses from each node
     * @param failures list of failures from each node.
     */
    public KNNCircuitBreakerTrippedResponse(
        ClusterName clusterName,
        List<KNNCircuitBreakerTrippedNodeResponse> nodes,
        List<FailedNodeException> failures
    ) {
        super(clusterName, nodes, failures);
        this.isTripped = checkIfTripped(nodes);
    }

    /**
     * Constructor.
     *
     * @param in input stream
     * @throws IOException thrown when input stream cannot be read
     */
    public KNNCircuitBreakerTrippedResponse(StreamInput in) throws IOException {
        super(new ClusterName(in), in.readList(KNNCircuitBreakerTrippedNodeResponse::new), in.readList(FailedNodeException::new));
        this.isTripped = in.readBoolean();
    }

    @Override
    protected List<KNNCircuitBreakerTrippedNodeResponse> readNodesFrom(StreamInput in) throws IOException {
        return in.readList(KNNCircuitBreakerTrippedNodeResponse::new);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeBoolean(isTripped);
    }

    @Override
    protected void writeNodesTo(StreamOutput out, List<KNNCircuitBreakerTrippedNodeResponse> nodes) throws IOException {
        out.writeList(nodes);
    }

    private boolean checkIfTripped(List<KNNCircuitBreakerTrippedNodeResponse> nodeResponses) {
        for (KNNCircuitBreakerTrippedNodeResponse nodeResponse : nodeResponses) {
            if (nodeResponse.isTripped()) {
                return true;
            }
        }
        return false;
    }
}
