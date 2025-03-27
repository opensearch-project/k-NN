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

package org.opensearch.knn.plugin.rest;

import com.google.common.collect.ImmutableList;
import org.apache.commons.lang.StringUtils;
import org.opensearch.transport.client.node.NodeClient;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.GetModelAction;
import org.opensearch.knn.plugin.transport.GetModelRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * Rest Handler for get model api endpoint.
 */
public class RestGetModelHandler extends BaseRestHandler {

    private final static String NAME = "knn_get_model_action";

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList.of(
            new Route(RestRequest.Method.GET, String.format(Locale.ROOT, "%s/%s/{%s}", KNNPlugin.KNN_BASE_URI, MODELS, MODEL_ID))
        );
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) throws IOException {
        String modelID = restRequest.param(MODEL_ID);
        if (StringUtils.isBlank(modelID)) {
            throw new IllegalArgumentException("model ID cannot be empty");
        }

        GetModelRequest getModelRequest = new GetModelRequest(modelID);
        return channel -> client.execute(GetModelAction.INSTANCE, getModelRequest, new RestToXContentListener<>(channel));
    }
}
