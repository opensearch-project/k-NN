/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.rest;

import lombok.AllArgsConstructor;
import org.apache.commons.lang.StringUtils;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsRequest;
import com.google.common.collect.ImmutableList;
import org.opensearch.transport.client.node.NodeClient;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestActions;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Resthandler for stats api endpoint. The user has the ability to get all stats from
 * all nodes or select stats from specific nodes.
 */
@AllArgsConstructor
public class RestKNNStatsHandler extends BaseRestHandler {
    private static final String NAME = "knn_stats_action";

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<Route> routes() {
        return getStatsRoutes();
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
        // From restrequest, create a knnStatsRequest
        KNNStatsRequest knnStatsRequest = getRequest(request);

        return channel -> client.execute(KNNStatsAction.INSTANCE, knnStatsRequest, new RestActions.NodesResponseRestListener<>(channel));
    }

    /**
     * Creates a KNNStatsRequest from a RestRequest
     *
     * @param request Rest request
     * @return KNNStatsRequest
     */
    private KNNStatsRequest getRequest(RestRequest request) {
        // parse the nodes the user wants to query
        String[] nodeIdsArr = null;
        String nodesIdsStr = request.param("nodeId");
        if (StringUtils.isNotEmpty(nodesIdsStr)) {
            nodeIdsArr = nodesIdsStr.split(",");
        }

        KNNStatsRequest knnStatsRequest = new KNNStatsRequest(nodeIdsArr);
        knnStatsRequest.timeout(request.param("timeout"));

        // parse the stats the customer wants to see
        Set<String> statsSet = null;
        String statsStr = request.param("stat");
        if (StringUtils.isNotEmpty(statsStr)) {
            statsSet = new HashSet<>(Arrays.asList(statsStr.split(",")));
        }

        if (statsSet == null) {
            knnStatsRequest.all();
        } else if (statsSet.size() == 1 && statsSet.contains("_all")) {
            knnStatsRequest.all();
        } else if (statsSet.contains(KNNStatsRequest.ALL_STATS_KEY)) {
            throw new IllegalArgumentException("Request " + request.path() + " contains _all and individual stats");
        } else {
            Set<String> invalidStats = new TreeSet<>();
            for (String stat : statsSet) {
                if (!knnStatsRequest.addStat(stat)) {
                    invalidStats.add(stat);
                }
            }

            if (!invalidStats.isEmpty()) {
                throw new IllegalArgumentException(unrecognized(request, invalidStats, knnStatsRequest.getStatsToBeRetrieved(), "stat"));
            }

        }
        return knnStatsRequest;
    }

    private static final List<Route> STATS_ROUTES = ImmutableList.of(
        new Route(RestRequest.Method.GET, KNNPlugin.KNN_BASE_URI + "/{nodeId}/stats/"),
        new Route(RestRequest.Method.GET, KNNPlugin.KNN_BASE_URI + "/{nodeId}/stats/{stat}"),
        new Route(RestRequest.Method.GET, KNNPlugin.KNN_BASE_URI + "/stats/"),
        new Route(RestRequest.Method.GET, KNNPlugin.KNN_BASE_URI + "/stats/{stat}")
    );

    private List<Route> getStatsRoutes() {
        return STATS_ROUTES;
    }
}
