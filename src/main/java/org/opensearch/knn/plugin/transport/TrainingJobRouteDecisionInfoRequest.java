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
import org.opensearch.common.io.stream.StreamInput;

import java.io.IOException;

/**
 * Request to get training job route decision info from some/all nodes in the cluster. Caller is
 * responsible for filtering out non-data nodes.
 */
public class TrainingJobRouteDecisionInfoRequest extends BaseNodesRequest<TrainingJobRouteDecisionInfoRequest> {

    /**
     * Constructor
     *
     * @param in input stream
     * @throws java.io.IOException in case of I/O errors
     */
    public TrainingJobRouteDecisionInfoRequest(StreamInput in) throws IOException {
        super(in);
    }

    /**
     * Constructor
     *
     * @param nodeIds NodeIDs from which to retrieve job route decision info
     */
    public TrainingJobRouteDecisionInfoRequest(String... nodeIds) {
        super(nodeIds);
    }
}
