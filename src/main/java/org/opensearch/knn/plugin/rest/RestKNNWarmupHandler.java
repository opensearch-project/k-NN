/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.rest;

import org.apache.commons.lang3.StringUtils;
import org.opensearch.knn.common.exception.KNNInvalidIndicesException;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.KNNWarmupAction;
import org.opensearch.knn.plugin.transport.KNNWarmupRequest;
import com.google.common.collect.ImmutableList;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.transport.client.node.NodeClient;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.index.Index;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestController;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.action.support.IndicesOptions.strictExpandOpen;

/**
 * RestHandler for k-NN index warmup API. API provides the ability for a user to load specific indices' k-NN graphs
 * into memory.
 */
public class RestKNNWarmupHandler extends BaseRestHandler {
    private static final Logger logger = LogManager.getLogger(RestKNNWarmupHandler.class);
    private static final String URL_PATH = "/warmup/{index}";
    public static String NAME = "knn_warmup_action";
    private IndexNameExpressionResolver indexNameExpressionResolver;
    private ClusterService clusterService;

    public RestKNNWarmupHandler(
        Settings settings,
        RestController controller,
        ClusterService clusterService,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        this.clusterService = clusterService;
        this.indexNameExpressionResolver = indexNameExpressionResolver;
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList.of(new Route(RestRequest.Method.GET, KNNPlugin.KNN_BASE_URI + URL_PATH));
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
        KNNWarmupRequest knnWarmupRequest = createKNNWarmupRequest(request);
        logger.info("[KNN] Warmup started for the following indices: " + String.join(",", knnWarmupRequest.indices()));
        return channel -> client.execute(KNNWarmupAction.INSTANCE, knnWarmupRequest, new RestToXContentListener<>(channel));
    }

    private KNNWarmupRequest createKNNWarmupRequest(RestRequest request) {
        String[] indexNames = StringUtils.split(request.param("index"), ",");
        Index[] indices = indexNameExpressionResolver.concreteIndices(clusterService.state(), strictExpandOpen(), indexNames);
        List<String> invalidIndexNames = new ArrayList<>();

        Arrays.stream(indices).forEach(index -> {
            if (!"true".equals(clusterService.state().metadata().getIndexSafe(index).getSettings().get(KNN_INDEX))) {
                invalidIndexNames.add(index.getName());
            }
        });

        if (invalidIndexNames.size() != 0) {
            throw new KNNInvalidIndicesException(
                invalidIndexNames,
                "Warm up request rejected. One or more indices have 'index.knn' set to false."
            );
        }

        return new KNNWarmupRequest(indexNames);
    }
}
