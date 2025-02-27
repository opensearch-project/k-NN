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
import org.opensearch.action.support.nodes.BaseNodeResponse;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;

/**
 * Node response for if KNNCircuitBreaker is tripped or not
 */
@Getter
public class KNNCircuitBreakerTrippedNodeResponse extends BaseNodeResponse {

    private final boolean isTripped;

    /**
     * Constructor from Stream.
     *
     * @param in stream input
     * @throws IOException thrown when unable to read from stream
     */
    public KNNCircuitBreakerTrippedNodeResponse(StreamInput in) throws IOException {
        super(in);
        isTripped = in.readBoolean();
    }

    /**
     * Constructor
     *
     * @param node node
     */
    public KNNCircuitBreakerTrippedNodeResponse(DiscoveryNode node, boolean isTripped) {
        super(node);
        this.isTripped = isTripped;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeBoolean(isTripped);
    }
}
