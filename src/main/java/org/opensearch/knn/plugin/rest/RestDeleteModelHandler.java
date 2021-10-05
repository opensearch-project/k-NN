package org.opensearch.knn.plugin.rest;

import com.google.common.collect.ImmutableList;
import org.opensearch.client.node.NodeClient;
import org.opensearch.common.Strings;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.DeleteModelAction;
import org.opensearch.knn.plugin.transport.DeleteModelRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.rest.action.admin.cluster.RestNodesUsageAction;

import java.util.List;
import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

public class RestDeleteModelHandler extends BaseRestHandler {

    public static final String NAME = "knn_delete_model_action";

    /**
     * @return the name of RestDeleteModelHandler.This is used in the response to the
     * {@link RestNodesUsageAction}.
     */
    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
            .of(
                new Route(
                    RestRequest.Method.DELETE,
                    String.format(Locale.ROOT, "%s/%s/{%s}", KNNPlugin.KNN_BASE_URI, MODELS, MODEL_ID)
                )
            );
    }

    /**
     * Prepare the request for deleting model.
     *
     * @param request the request to execute
     * @param client  client for executing actions on the local node
     * @return the action to execute
     */
    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
        String modelID = request.param(MODEL_ID);
        if (!Strings.hasText(modelID)) {
            throw new IllegalArgumentException("model ID cannot be empty");
        }
        DeleteModelRequest deleteModelRequest = new DeleteModelRequest(modelID);
        return channel -> client.execute(DeleteModelAction.INSTANCE, deleteModelRequest, new RestToXContentListener<>(channel));
    }
}
