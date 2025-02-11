/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.rest;

import com.google.common.collect.ImmutableList;
import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.transport.client.node.NodeClient;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.common.Strings;
import org.opensearch.core.index.Index;
import org.opensearch.knn.common.exception.KNNInvalidIndicesException;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.ClearCacheAction;
import org.opensearch.knn.plugin.transport.ClearCacheRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import static org.opensearch.action.support.IndicesOptions.strictExpandOpen;
import static org.opensearch.knn.common.KNNConstants.CLEAR_CACHE;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

/**
 * RestHandler for k-NN Clear Cache API. API provides the ability for a user to evict those indices from Cache.
 */
@AllArgsConstructor
@Log4j2
public class RestClearCacheHandler extends BaseRestHandler {
    private static final String INDEX = "index";
    public static String NAME = "knn_clear_cache_action";
    private final ClusterService clusterService;
    private final IndexNameExpressionResolver indexNameExpressionResolver;

    /**
     * @return name of Clear Cache API action
     */
    @Override
    public String getName() {
        return NAME;
    }

    /**
     * @return Immutable List of Clear Cache API endpoint
     */
    @Override
    public List<Route> routes() {
        return ImmutableList.of(
            new Route(RestRequest.Method.POST, String.format(Locale.ROOT, "%s/%s/{%s}", KNNPlugin.KNN_BASE_URI, CLEAR_CACHE, INDEX))
        );
    }

    /**
     * @param request RestRequest
     * @param client NodeClient
     * @return RestChannelConsumer
     */
    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
        ClearCacheRequest clearCacheRequest = createClearCacheRequest(request);
        log.info("[KNN] ClearCache started for the following indices: [{}]", String.join(",", clearCacheRequest.indices()));
        return channel -> client.execute(ClearCacheAction.INSTANCE, clearCacheRequest, new RestToXContentListener<>(channel));
    }

    // Create a clear cache request by processing the rest request and validating the indices
    private ClearCacheRequest createClearCacheRequest(RestRequest request) {
        String[] indexNames = Strings.splitStringByCommaToArray(request.param("index"));
        Index[] indices = indexNameExpressionResolver.concreteIndices(clusterService.state(), strictExpandOpen(), indexNames);
        validateIndices(indices);

        return new ClearCacheRequest(indexNames);
    }

    // Validate if the given indices are k-NN indices or not. If there are any invalid indices,
    // the request is rejected and an exception is thrown.
    private void validateIndices(Index[] indices) {
        List<String> invalidIndexNames = Arrays.stream(indices)
            .filter(index -> !"true".equals(clusterService.state().metadata().getIndexSafe(index).getSettings().get(KNN_INDEX)))
            .map(Index::getName)
            .collect(Collectors.toList());

        if (!invalidIndexNames.isEmpty()) {
            throw new KNNInvalidIndicesException(
                invalidIndexNames,
                "ClearCache request rejected. One or more indices have 'index.knn' set to false."
            );
        }
    }
}
