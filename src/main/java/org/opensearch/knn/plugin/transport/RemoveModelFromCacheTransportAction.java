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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

/**
 * Transport action to remove models from some or all nodes in the clusters caches
 */
public class RemoveModelFromCacheTransportAction extends TransportNodesAction<
    RemoveModelFromCacheRequest,
    RemoveModelFromCacheResponse,
    RemoveModelFromCacheNodeRequest,
    RemoveModelFromCacheNodeResponse> {

    private static Logger logger = LogManager.getLogger(RemoveModelFromCacheTransportAction.class);

    @Inject
    public RemoveModelFromCacheTransportAction(
        ThreadPool threadPool,
        ClusterService clusterService,
        TransportService transportService,
        ActionFilters actionFilters
    ) {
        super(
            RemoveModelFromCacheAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            RemoveModelFromCacheRequest::new,
            RemoveModelFromCacheNodeRequest::new,
            ThreadPool.Names.SAME,
            RemoveModelFromCacheNodeResponse.class
        );
    }

    @Override
    protected RemoveModelFromCacheResponse newResponse(
        RemoveModelFromCacheRequest nodesRequest,
        List<RemoveModelFromCacheNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        return new RemoveModelFromCacheResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected RemoveModelFromCacheNodeRequest newNodeRequest(RemoveModelFromCacheRequest request) {
        return new RemoveModelFromCacheNodeRequest(request.getModelId());
    }

    @Override
    protected RemoveModelFromCacheNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new RemoveModelFromCacheNodeResponse(in);
    }

    @Override
    protected RemoveModelFromCacheNodeResponse nodeOperation(RemoveModelFromCacheNodeRequest nodeRequest) {
        logger.debug("[KNN] Removing model \"" + nodeRequest.getModelId() + "\" on node \"" + clusterService.localNode().getId() + ".");
        ModelCache.getInstance().remove(nodeRequest.getModelId());
        return new RemoveModelFromCacheNodeResponse(clusterService.localNode());
    }
}
