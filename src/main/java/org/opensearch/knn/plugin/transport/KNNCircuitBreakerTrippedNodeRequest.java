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

import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.transport.TransportRequest;

import java.io.IOException;

/**
 * Node request to detect if the circuit breaker has been tripped
 */
public class KNNCircuitBreakerTrippedNodeRequest extends TransportRequest {

    /**
     * Constructor.
     */
    public KNNCircuitBreakerTrippedNodeRequest() {
        super();
    }

    /**
     * Constructor from stream
     *
     * @param in input stream
     * @throws IOException thrown when reading from stream fails
     */
    public KNNCircuitBreakerTrippedNodeRequest(StreamInput in) throws IOException {
        super(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
    }
}
