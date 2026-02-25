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
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;
import java.util.List;

/**
 * Aggregated RemoveModelFromCacheNodeResponse's for all nodes the request was sent to.
 */
public class RemoveModelFromCacheResponse extends BaseNodesResponse<RemoveModelFromCacheNodeResponse> {

    /**
     * Constructor.
     *
     * @param clusterName cluster's name
     * @param nodes list of responses from each node
     * @param failures list of failures from each node.
     */
    public RemoveModelFromCacheResponse(
        ClusterName clusterName,
        List<RemoveModelFromCacheNodeResponse> nodes,
        List<FailedNodeException> failures
    ) {
        super(clusterName, nodes, failures);
    }

    /**
     * Constructor.
     *
     * @param in input stream
     * @throws IOException thrown when input stream cannot be read
     */
    public RemoveModelFromCacheResponse(StreamInput in) throws IOException {
        super(new ClusterName(in), in.readList(RemoveModelFromCacheNodeResponse::new), in.readList(FailedNodeException::new));
    }

    @Override
    protected List<RemoveModelFromCacheNodeResponse> readNodesFrom(StreamInput in) throws IOException {
        return in.readList(RemoveModelFromCacheNodeResponse::new);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
    }

    @Override
    protected void writeNodesTo(StreamOutput out, List<RemoveModelFromCacheNodeResponse> nodes) throws IOException {
        out.writeList(nodes);
    }
}
