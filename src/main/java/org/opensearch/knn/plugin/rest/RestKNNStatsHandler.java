/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.rest;

import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.stats.KNNStats;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsRequest;
import com.google.common.collect.ImmutableList;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.client.node.NodeClient;
import org.opensearch.common.Strings;
import org.opensearch.common.settings.Settings;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestController;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestActions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

/**
 * Resthandler for stats api endpoint. The user has the ability to get all stats from
 * all nodes or select stats from specific nodes.
 */
public class RestKNNStatsHandler extends BaseRestHandler {

    private static final Logger LOG = LogManager.getLogger(RestKNNStatsHandler.class);
    private static final String NAME = "knn_stats_action";
    private KNNStats knnStats;

    /**
     * Constructor
     *
     * @param settings Settings
     * @param controller Rest Controller
     * @param knnStats KNNStats
     */
    public RestKNNStatsHandler(Settings settings, RestController controller, KNNStats knnStats) {
        this.knnStats = knnStats;
    }

    @Override
    public String getName() {
        return NAME;
    }


    private List<String> getStatsPath() {
        List<String> statsPath = new ArrayList<>();
        statsPath.add("/{nodeId}/stats/");
        statsPath.add("/{nodeId}/stats/{stat}");
        statsPath.add("/stats/");
        statsPath.add("/stats/{stat}");
        return statsPath;
    }

    private Map<String, String> getUrlPathByLegacyUrlPathMap() {
        return getStatsPath().stream().collect(
            Collectors.toMap(path -> KNNPlugin.LEGACY_KNN_BASE_URI + path, path -> KNNPlugin.KNN_BASE_URI + path)
        );
    }

    @Override
    public List<Route> routes() {
      return ImmutableList.of();
    }

    @Override
    public List<ReplacedRoute> replacedRoutes() {
        return getUrlPathByLegacyUrlPathMap().entrySet().stream().map(
            e -> new ReplacedRoute(RestRequest.Method.GET, e.getValue(), RestRequest.Method.GET, e.getKey())
        ).collect(Collectors.toList());
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
        if (!Strings.isEmpty(nodesIdsStr)) {
            nodeIdsArr = nodesIdsStr.split(",");
        }

        KNNStatsRequest knnStatsRequest = new KNNStatsRequest(knnStats.getStats().keySet(), nodeIdsArr);
        knnStatsRequest.timeout(request.param("timeout"));

        // parse the stats the customer wants to see
        Set<String> statsSet = null;
        String statsStr = request.param("stat");
        if (!Strings.isEmpty(statsStr)) {
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
                throw new IllegalArgumentException(unrecognized(request, invalidStats,
                        knnStatsRequest.getStatsToBeRetrieved(), "stat"));
            }

        }
        return knnStatsRequest;
    }
}
