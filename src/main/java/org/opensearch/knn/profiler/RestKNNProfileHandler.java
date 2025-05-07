/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import com.google.common.collect.ImmutableList;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.KNNProfileAction;
import org.opensearch.knn.plugin.transport.KNNProfileRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;

import java.util.List;

/**
 * RestHandler for k-NN index warmup API. API provides the ability for a user to load specific indices' k-NN graphs
 * into memory.
 */
public class RestKNNProfileHandler extends BaseRestHandler {
    private static final Logger logger = LogManager.getLogger(RestKNNProfileHandler.class);
    private static final String URL_PATH = "/profile/{index}/{field}";
    public static String NAME = "knn_profile_action";
    private IndexNameExpressionResolver indexNameExpressionResolver;
    private ClusterService clusterService;

    public RestKNNProfileHandler() {}

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
        KNNProfileRequest knnProfileRequest = createKNNProfileRequest(request);
        return channel -> client.execute(KNNProfileAction.INSTANCE, knnProfileRequest, new RestToXContentListener<>(channel));
    }

    private KNNProfileRequest createKNNProfileRequest(RestRequest request) {
        String indexName = request.param("index");
        String fieldName = request.param("field");

        return new KNNProfileRequest(indexName, fieldName);
    }
}
