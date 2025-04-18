/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.apache.commons.lang.StringUtils;
import org.opensearch.knn.common.exception.KNNInvalidIndicesException;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.KNNProfileAction;
import org.opensearch.knn.plugin.transport.KNNProfileRequest;
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.action.support.IndicesOptions.strictExpandOpen;

/**
 * RestHandler for k-NN index profile API. API provides the ability for a user to get statistical information
 * about vector dimensions in specific indices.
 */
public class RestKNNProfileHandler extends BaseRestHandler {
    private static final Logger logger = LogManager.getLogger(RestKNNProfileHandler.class);
    private static final String URL_PATH = "/profile/{index}";
    public static String NAME = "knn_profile_action";
    private IndexNameExpressionResolver indexNameExpressionResolver;
    private ClusterService clusterService;

    public RestKNNProfileHandler(
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
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        KNNProfileRequest knnProfileRequest = createKNNProfileRequest(request);
        logger.info(
            "[KNN] Profile started for the following indices: {} and field: {}",
            String.join(",", knnProfileRequest.indices()),
            knnProfileRequest.getFieldName()
        );
        return channel -> client.execute(KNNProfileAction.INSTANCE, knnProfileRequest, new RestToXContentListener<>(channel));
    }

    private KNNProfileRequest createKNNProfileRequest(RestRequest request) throws IOException {
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
                "Profile request rejected. One or more indices have 'index.knn' set to false."
            );
        }

        KNNProfileRequest profileRequest = new KNNProfileRequest(indexNames);

        String fieldName = request.param("field_name", "my_vector_field");
        profileRequest.setFieldName(fieldName);

        return profileRequest;
    }
}
