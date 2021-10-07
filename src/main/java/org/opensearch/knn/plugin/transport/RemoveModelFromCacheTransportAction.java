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
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

public class RemoveModelFromCacheTransportAction extends
        TransportNodesAction<RemoveModelFromCacheRequest, RemoveModelFromCacheResponse,
                RemoveModelFromCacheNodeRequest, RemoveFromCacheNodeResponse> {


    protected RemoveModelFromCacheTransportAction(String actionName, ThreadPool threadPool, ClusterService clusterService, TransportService transportService, ActionFilters actionFilters, Writeable.Reader<RemoveModelFromCacheRequest> request, Writeable.Reader<RemoveModelFromCacheNodeRequest> nodeRequest, String nodeExecutor, String finalExecutor, Class<RemoveFromCacheNodeResponse> removeFromCacheNodeResponseClass) {
        super(actionName, threadPool, clusterService, transportService, actionFilters, request, nodeRequest, nodeExecutor, finalExecutor, removeFromCacheNodeResponseClass);
    }

    @Override
    protected RemoveModelFromCacheResponse newResponse(RemoveModelFromCacheRequest nodesRequest, List<RemoveFromCacheNodeResponse> list, List<FailedNodeException> list1) {
        return null;
    }

    @Override
    protected RemoveModelFromCacheNodeRequest newNodeRequest(RemoveModelFromCacheRequest nodesRequest) {
        return null;
    }

    @Override
    protected RemoveFromCacheNodeResponse newNodeResponse(StreamInput streamInput) throws IOException {
        return null;
    }

    @Override
    protected RemoveFromCacheNodeResponse nodeOperation(RemoveModelFromCacheNodeRequest nodeRequest) {
        return null;
    }
}
