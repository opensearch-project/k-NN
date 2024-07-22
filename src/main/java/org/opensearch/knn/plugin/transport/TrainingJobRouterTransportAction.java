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

import org.apache.commons.lang.StringUtils;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.ActionListenerResponseHandler;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.ValidationException;
import org.opensearch.common.inject.Inject;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportRequestOptions;
import org.opensearch.transport.TransportService;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.search.internal.SearchContext.DEFAULT_TERMINATE_AFTER;

/**
 * Sends training request to appropriate node
 */
public class TrainingJobRouterTransportAction extends HandledTransportAction<TrainingModelRequest, TrainingModelResponse> {

    private final TransportService transportService;
    private final ClusterService clusterService;
    private final Client client;

    @Inject
    public TrainingJobRouterTransportAction(
        TransportService transportService,
        ActionFilters actionFilters,
        ClusterService clusterService,
        Client client
    ) {
        super(TrainingJobRouterAction.NAME, transportService, actionFilters, TrainingModelRequest::new);
        this.clusterService = clusterService;
        this.client = client;
        this.transportService = transportService;
    }

    @Override
    protected void doExecute(Task task, TrainingModelRequest request, ActionListener<TrainingModelResponse> listener) {
        // Get the size of the training request and then route the request. We get/set this here, as opposed to in
        // TrainingModelTransportAction, because in the future, we may want to use size to factor into our routing
        // decision.
        getTrainingIndexSizeInKB(request, ActionListener.wrap(size -> {
            request.setTrainingDataSizeInKB(size);
            routeRequest(request, listener);
        }, listener::onFailure));
    }

    protected void routeRequest(TrainingModelRequest request, ActionListener<TrainingModelResponse> listener) {
        // Pick a node and then use the transport service to forward the request
        client.execute(
            TrainingJobRouteDecisionInfoAction.INSTANCE,
            new TrainingJobRouteDecisionInfoRequest(),
            ActionListener.wrap(response -> {
                DiscoveryNode node = selectNode(request.getPreferredNodeId(), response);

                if (node == null) {
                    ValidationException exception = new ValidationException();
                    exception.addValidationError("Cluster does not have capacity to train");
                    listener.onFailure(exception);
                    return;
                }

                transportService.sendRequest(
                    node,
                    TrainingModelAction.NAME,
                    request,
                    TransportRequestOptions.EMPTY,
                    new ActionListenerResponseHandler<>(listener, TrainingModelResponse::new)
                );
            }, listener::onFailure)
        );
    }

    protected DiscoveryNode selectNode(String preferredNode, TrainingJobRouteDecisionInfoResponse jobInfo) {

        DiscoveryNode selectedNode = null;

        Map<String, DiscoveryNode> eligibleNodes = clusterService.state().nodes().getDataNodes();
        DiscoveryNode currentNode;

        for (TrainingJobRouteDecisionInfoNodeResponse response : jobInfo.getNodes()) {
            currentNode = response.getNode();

            if (!eligibleNodes.containsKey(currentNode.getId())) {
                continue;
            }

            if (response.getTrainingJobCount() < 1) {
                selectedNode = currentNode;
                // Return right away if the user didnt pass a preferred node or this is the preferred node
                if (StringUtils.isEmpty(preferredNode) || selectedNode.getId().equals(preferredNode)) {
                    return selectedNode;
                }
            }
        }

        return selectedNode;
    }

    protected void getTrainingIndexSizeInKB(TrainingModelRequest trainingModelRequest, ActionListener<Integer> listener) {
        // For this function, I referred to the rest count action: https://github.com/opensearch-project/OpenSearch/
        // blob/main/server/src/main/java/org/opensearch/rest/action/search/RestCountAction.java
        SearchRequest countRequest = new SearchRequest(trainingModelRequest.getTrainingIndex());
        SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder().size(0).trackTotalHits(true);
        countRequest.source(searchSourceBuilder);
        searchSourceBuilder.terminateAfter(DEFAULT_TERMINATE_AFTER);

        client.search(countRequest, ActionListener.wrap(searchResponse -> {
            long trainingVectors = searchResponse.getHits().getTotalHits().value;

            // If there are more docs in the index than what the user wants to use for training, take the min
            if (trainingModelRequest.getMaximumVectorCount() < trainingVectors) {
                trainingVectors = trainingModelRequest.getMaximumVectorCount();
            }

            listener.onResponse(
                estimateVectorSetSizeInKB(trainingVectors, trainingModelRequest.getDimension(), trainingModelRequest.getVectorDataType())
            );
        }, listener::onFailure));
    }

    /**
     * Estimates the size of a set of vectors in KB
     *
     * @param vectorCount number of vectors
     * @param dimension dimension of vectors
     * @return size estimate
     */
    public static int estimateVectorSetSizeInKB(long vectorCount, int dimension, VectorDataType vectorDataType) {
        switch (vectorDataType) {
            case BINARY:
                return Math.toIntExact(((Byte.BYTES * (dimension / 8) * vectorCount) / BYTES_PER_KILOBYTES) + 1L);
            case BYTE:
                return Math.toIntExact(((Byte.BYTES * dimension * vectorCount) / BYTES_PER_KILOBYTES) + 1L);
            default:
                return Math.toIntExact(((Float.BYTES * dimension * vectorCount) / BYTES_PER_KILOBYTES) + 1L);
        }
    }
}
