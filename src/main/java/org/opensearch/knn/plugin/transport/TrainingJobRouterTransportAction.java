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

import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionListenerResponseHandler;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.collect.ImmutableOpenMap;
import org.opensearch.common.inject.Inject;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportRequestOptions;
import org.opensearch.transport.TransportService;

import java.util.concurrent.RejectedExecutionException;

/**
 * Sends training request to appropriate node
 */
public class TrainingJobRouterTransportAction extends HandledTransportAction<TrainingModelRequest, TrainingModelResponse> {

    private final TransportService transportService;
    private final ClusterService clusterService;
    private final Client client;

    @Inject
    public TrainingJobRouterTransportAction(TransportService transportService,
                                            ActionFilters actionFilters,
                                            ClusterService clusterService, Client client) {
        super(TrainingJobRouterAction.NAME, transportService, actionFilters, TrainingModelRequest::new);
        this.clusterService = clusterService;
        this.client = client;
        this.transportService = transportService;
    }

    @Override
    protected void doExecute(Task task, TrainingModelRequest request,
                             ActionListener<TrainingModelResponse> listener) {
        // Pick a node and then use the transport service to forward the request
        client.execute(TrainingJobRouteDecisionInfoAction.INSTANCE, new TrainingJobRouteDecisionInfoRequest(),
                ActionListener.wrap(response -> {
                    DiscoveryNode node = selectNode(request.getPreferredNodeId(), response);

                    if (node == null) {
                        listener.onFailure(new RejectedExecutionException("Cluster does not have capacity to train"));
                        return;
                    }

                    transportService.sendRequest(node, TrainingModelAction.NAME, request, TransportRequestOptions.EMPTY,
                            new ActionListenerResponseHandler<>(listener, TrainingModelResponse::new));
                }, listener::onFailure));
    }

    protected DiscoveryNode selectNode(String preferredNode, TrainingJobRouteDecisionInfoResponse jobInfo) {

        DiscoveryNode selectedNode = null;

        ImmutableOpenMap<String, DiscoveryNode> eligibleNodes = clusterService.state().nodes().getDataNodes();
        DiscoveryNode currentNode;
        for (TrainingJobRouteDecisionInfoNodeResponse response : jobInfo.getNodes()) {
            currentNode = response.getNode();

            // If the node has already been selected and the current node's id is not preferred, skip
            if (selectedNode != null && !currentNode.getId().equals(preferredNode)) {
                continue;
            }

            if (response.getTrainingJobCount() < 1 && eligibleNodes.containsKey(currentNode.getId())) {
                selectedNode = currentNode;

                // Return right away if this is the preferred node
                if (selectedNode.getId().equals(preferredNode)) {
                    return selectedNode;
                }
            }
        }

        return selectedNode;
    }
}
