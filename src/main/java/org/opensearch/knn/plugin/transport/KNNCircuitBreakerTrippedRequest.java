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

import org.opensearch.action.support.nodes.BaseNodesRequest;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;

/**
 * Request to check if circuit breaker for cluster has been tripped
 */
public class KNNCircuitBreakerTrippedRequest extends BaseNodesRequest<KNNCircuitBreakerTrippedRequest> {

    /**
     * Constructor.
     *
     * @param nodeIds Id's of nodes
     */
    public KNNCircuitBreakerTrippedRequest(String... nodeIds) {
        super(nodeIds);
    }

    /**
     * Constructor.
     *
     * @param in input stream
     * @throws IOException thrown when reading input stream fails
     */
    public KNNCircuitBreakerTrippedRequest(StreamInput in) throws IOException {
        super(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
    }
}
