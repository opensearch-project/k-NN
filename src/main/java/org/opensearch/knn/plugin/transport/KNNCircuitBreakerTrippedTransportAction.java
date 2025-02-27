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

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.knn.index.KNNCircuitBreaker;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

/**
 * Transport action to check if the KNN CB is tripped on any of the nodes in the cluster.
 */
@Log4j2
public class KNNCircuitBreakerTrippedTransportAction extends TransportNodesAction<
    KNNCircuitBreakerTrippedRequest,
    KNNCircuitBreakerTrippedResponse,
    KNNCircuitBreakerTrippedNodeRequest,
    KNNCircuitBreakerTrippedNodeResponse> {

    @Inject
    public KNNCircuitBreakerTrippedTransportAction(
        ThreadPool threadPool,
        ClusterService clusterService,
        TransportService transportService,
        ActionFilters actionFilters
    ) {
        super(
            KNNCircuitBreakerTrippedAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            KNNCircuitBreakerTrippedRequest::new,
            KNNCircuitBreakerTrippedNodeRequest::new,
            ThreadPool.Names.SAME,
            KNNCircuitBreakerTrippedNodeResponse.class
        );
    }

    @Override
    protected KNNCircuitBreakerTrippedResponse newResponse(
        KNNCircuitBreakerTrippedRequest nodesRequest,
        List<KNNCircuitBreakerTrippedNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        return new KNNCircuitBreakerTrippedResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected KNNCircuitBreakerTrippedNodeRequest newNodeRequest(KNNCircuitBreakerTrippedRequest request) {
        return new KNNCircuitBreakerTrippedNodeRequest();
    }

    @Override
    protected KNNCircuitBreakerTrippedNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new KNNCircuitBreakerTrippedNodeResponse(in);
    }

    @Override
    protected KNNCircuitBreakerTrippedNodeResponse nodeOperation(KNNCircuitBreakerTrippedNodeRequest nodeRequest) {
        return new KNNCircuitBreakerTrippedNodeResponse(clusterService.localNode(), KNNCircuitBreaker.getInstance().isTripped());
    }
}
