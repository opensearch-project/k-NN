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
import org.opensearch.common.inject.Inject;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.knn.training.TrainingJobRunner;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

/**
 * Broadcasts request to collect training job route decision info from all nodes and aggregates it into a single
 * response.
 */
public class TrainingJobRouteDecisionInfoTransportAction extends TransportNodesAction<
    TrainingJobRouteDecisionInfoRequest,
    TrainingJobRouteDecisionInfoResponse,
    TrainingJobRouteDecisionInfoNodeRequest,
    TrainingJobRouteDecisionInfoNodeResponse> {
    /**
     * Constructor
     *
     * @param threadPool ThreadPool to use
     * @param clusterService ClusterService
     * @param transportService TransportService
     * @param actionFilters Action Filters
     */
    @Inject
    public TrainingJobRouteDecisionInfoTransportAction(
        ThreadPool threadPool,
        ClusterService clusterService,
        TransportService transportService,
        ActionFilters actionFilters
    ) {
        super(
            TrainingJobRouteDecisionInfoAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            TrainingJobRouteDecisionInfoRequest::new,
            TrainingJobRouteDecisionInfoNodeRequest::new,
            ThreadPool.Names.MANAGEMENT,
            TrainingJobRouteDecisionInfoNodeResponse.class
        );
    }

    @Override
    protected TrainingJobRouteDecisionInfoResponse newResponse(
        TrainingJobRouteDecisionInfoRequest request,
        List<TrainingJobRouteDecisionInfoNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        return new TrainingJobRouteDecisionInfoResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected TrainingJobRouteDecisionInfoNodeRequest newNodeRequest(TrainingJobRouteDecisionInfoRequest request) {
        return new TrainingJobRouteDecisionInfoNodeRequest();
    }

    @Override
    protected TrainingJobRouteDecisionInfoNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new TrainingJobRouteDecisionInfoNodeResponse(in);
    }

    @Override
    protected TrainingJobRouteDecisionInfoNodeResponse nodeOperation(TrainingJobRouteDecisionInfoNodeRequest request) {
        return new TrainingJobRouteDecisionInfoNodeResponse(clusterService.localNode(), TrainingJobRunner.getInstance().getJobCount());
    }
}
