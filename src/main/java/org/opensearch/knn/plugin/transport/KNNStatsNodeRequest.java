/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.transport.TransportRequest;

import java.io.IOException;

/**
 *  KNNStatsNodeRequest represents the request to an individual node
 */
public class KNNStatsNodeRequest extends TransportRequest {
    private KNNStatsRequest request;

    /**
     * Constructor
     */
    public KNNStatsNodeRequest() {
        super();
    }

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException in case of I/O errors
     */
    public KNNStatsNodeRequest(StreamInput in) throws IOException {
        super(in);
        request = new KNNStatsRequest(in);
    }

    /**
     * Constructor
     *
     * @param request KNNStatsRequest
     */
    public KNNStatsNodeRequest(KNNStatsRequest request) {
        this.request = request;
    }

    /**
     * Get KNNStatsRequest
     *
     * @return KNNStatsRequest for this node
     */
    public KNNStatsRequest getKNNStatsRequest() {
        return request;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        request.writeTo(out);
    }
}
