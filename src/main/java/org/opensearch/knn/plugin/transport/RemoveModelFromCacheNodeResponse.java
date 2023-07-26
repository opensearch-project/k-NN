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

import org.opensearch.action.support.nodes.BaseNodeResponse;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.core.common.io.stream.StreamInput;

import java.io.IOException;

/**
 * Default implementation of BaseNodeResponse. No additional information is needed from the nodes.
 */
public class RemoveModelFromCacheNodeResponse extends BaseNodeResponse {

    /**
     * Constructor from Stream.
     *
     * @param in stream input
     * @throws IOException thrown when unable to read from stream
     */
    public RemoveModelFromCacheNodeResponse(StreamInput in) throws IOException {
        super(in);
    }

    /**
     * Constructor
     *
     * @param node node
     */
    public RemoveModelFromCacheNodeResponse(DiscoveryNode node) {
        super(node);
    }
}
