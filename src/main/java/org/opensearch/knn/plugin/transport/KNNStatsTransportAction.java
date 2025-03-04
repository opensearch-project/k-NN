/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.plugin.stats.KNNNodeStatAggregation;
import org.opensearch.knn.plugin.stats.KNNStats;

import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 *  KNNStatsTransportAction contains the logic to extract the stats from the nodes
 */
public class KNNStatsTransportAction extends TransportNodesAction<
    KNNStatsRequest,
    KNNStatsResponse,
    KNNStatsNodeRequest,
    KNNStatsNodeResponse> {

    private final KNNStats knnStats;
    private final Client client;

    /**
     * Constructor
     *
     * @param threadPool ThreadPool to use
     * @param clusterService ClusterService
     * @param transportService TransportService
     * @param actionFilters Action Filters
     * @param knnStats KNNStats object
     */
    @Inject
    public KNNStatsTransportAction(
        ThreadPool threadPool,
        ClusterService clusterService,
        TransportService transportService,
        ActionFilters actionFilters,
        KNNStats knnStats,
        Client client
    ) {
        super(
            KNNStatsAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            KNNStatsRequest::new,
            KNNStatsNodeRequest::new,
            ThreadPool.Names.MANAGEMENT,
            KNNStatsNodeResponse.class
        );
        this.knnStats = knnStats;
        this.client = client;
    }

    @Override
    protected void doExecute(Task task, KNNStatsRequest request, ActionListener<KNNStatsResponse> listener) {
        // Some cluster stats may need node stats from all nodes. So, we may have to send out a request before
        // in order to create this aggregation. For example, the cluster level circuit breaker needs to get the
        // cache capacities from each node. If any are reached, it sets it to true. This is inefficient, but, stats
        // api does not need to be called frequently, so I dont see it causing any issues.
        ActionListener<Void> contextListener = ActionListener.wrap(none -> super.doExecute(task, request, listener), listener::onFailure);
        Set<String> statsToBeRetrieved = request.getStatsToBeRetrieved();
        Set<String> dependentStats = new HashSet<>();
        for (String statName : knnStats.getClusterStats().keySet()) {
            if (statsToBeRetrieved.contains(statName)) {
                dependentStats.addAll(knnStats.getClusterStats().get(statName).dependentNodeStats());
            }
        }

        if (dependentStats.stream().anyMatch(stat -> knnStats.getClusterStats().containsKey(stat))) {
            throw new IllegalStateException("Cluster stats cannot depend on other cluster stats.");
        }

        if (dependentStats.isEmpty() == false) {
            KNNStatsRequest knnStatsRequest = new KNNStatsRequest();
            // Add the stats makes sure that we dont recurse infinitely.
            dependentStats.forEach(knnStatsRequest::addStat);
            client.execute(KNNStatsAction.INSTANCE, knnStatsRequest, ActionListener.wrap(knnStatsResponse -> {
                request.setAggregation(new KNNNodeStatAggregation(knnStatsResponse.getNodes()));
                contextListener.onResponse(null);
            }, contextListener::onFailure));
        } else {
            contextListener.onResponse(null);
        }
    }

    @Override
    protected KNNStatsResponse newResponse(
        KNNStatsRequest request,
        List<KNNStatsNodeResponse> responses,
        List<FailedNodeException> failures
    ) {

        Map<String, Object> clusterStats = new HashMap<>();
        Set<String> statsToBeRetrieved = request.getStatsToBeRetrieved();

        for (String statName : knnStats.getClusterStats().keySet()) {
            if (statsToBeRetrieved.contains(statName)) {
                clusterStats.put(statName, knnStats.getStats().get(statName).getValue(request.getAggregation()));
            }
        }

        return new KNNStatsResponse(clusterService.getClusterName(), responses, failures, clusterStats);
    }

    @Override
    protected KNNStatsNodeRequest newNodeRequest(KNNStatsRequest request) {
        return new KNNStatsNodeRequest(request);
    }

    @Override
    protected KNNStatsNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new KNNStatsNodeResponse(in);
    }

    @Override
    protected KNNStatsNodeResponse nodeOperation(KNNStatsNodeRequest request) {
        return createKNNStatsNodeResponse(request.getKNNStatsRequest());
    }

    private KNNStatsNodeResponse createKNNStatsNodeResponse(KNNStatsRequest knnStatsRequest) {
        Map<String, Object> statValues = new HashMap<>();
        Set<String> statsToBeRetrieved = knnStatsRequest.getStatsToBeRetrieved();

        for (String statName : knnStats.getNodeStats().keySet()) {
            if (statsToBeRetrieved.contains(statName)) {
                statValues.put(statName, knnStats.getStats().get(statName).getValue());
            }
        }

        return new KNNStatsNodeResponse(clusterService.localNode(), statValues);
    }
}
