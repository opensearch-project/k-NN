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
import org.opensearch.transport.TransportRequest;

import java.io.IOException;

/**
 * Request sent to each node to gather info that will be used to determine where to route a job. Right now,
 * this is fairly simple. However, in the future, we could add different filter parameters here.
 */
public class TrainingJobRouteDecisionInfoNodeRequest extends TransportRequest {

    /**
     * Constructor
     */
    public TrainingJobRouteDecisionInfoNodeRequest() {
        super();
    }

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException in case of I/O errors
     */
    public TrainingJobRouteDecisionInfoNodeRequest(StreamInput in) throws IOException {
        super(in);
    }
}
