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
import org.opensearch.client.node.NodeClient;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.TrainingJobRouterAction;
import org.opensearch.knn.plugin.transport.TrainingModelRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.MAX_VECTOR_COUNT_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.SEARCH_SIZE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;

/**
 * Rest Handler for get model api endpoint.
 */
public class RestTrainModelHandler extends BaseRestHandler {

    private final static String NAME = "knn_train_model_action";

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
                .of(
                        new Route(
                                RestRequest.Method.POST,
                                String.format(Locale.ROOT, "%s/%s/{%s}/_train", KNNPlugin.KNN_BASE_URI, MODELS,
                                        MODEL_ID)
                        ),
                        new Route(
                                RestRequest.Method.POST,
                                String.format(Locale.ROOT, "%s/%s/_train", KNNPlugin.KNN_BASE_URI, MODELS)
                        )
                );
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) throws IOException {
        TrainingModelRequest trainingModelRequest = createTransportRequest(restRequest);

        return channel -> client.execute(TrainingJobRouterAction.INSTANCE, trainingModelRequest,
                new RestToXContentListener<>(channel));
    }

    private TrainingModelRequest createTransportRequest(RestRequest restRequest) throws IOException {
        // Parse query params
        String modelId = restRequest.param(MODEL_ID);
        String preferredNodeId = restRequest.param("preference");

        // Parse request body
        XContentParser parser = restRequest.contentParser();
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);

        KNNMethodContext knnMethodContext = null;
        int dimension = -1;
        String trainingIndex = null;
        String trainingField = null;
        String description = null;

        int maximumVectorCount = -1;
        int searchSize = -1;

        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            if (TRAIN_INDEX_PARAMETER.equals(fieldName) && trainingIndex == null) {
                trainingIndex = parser.text();
            } else if (TRAIN_FIELD_PARAMETER.equals(fieldName) && trainingField == null) {
                trainingField = parser.text();
            } else if (KNN_METHOD.equals(fieldName) && knnMethodContext == null) {
                knnMethodContext = KNNMethodContext.parse(parser.map());
            } else if (DIMENSION.equals(fieldName) && dimension == -1) {
                dimension = parser.intValue();
            } else if (MAX_VECTOR_COUNT_PARAMETER.equals(fieldName) && maximumVectorCount == -1) {
                maximumVectorCount = parser.intValue();
            } else if (SEARCH_SIZE_PARAMETER.equals(fieldName) && searchSize == -1) {
                searchSize = parser.intValue();
            } else if (MODEL_DESCRIPTION.equals(fieldName) && description == null) {
                description = parser.text();
            } else {
                throw new IllegalArgumentException("Unable to parse token \"" + fieldName + "\" either because it " +
                        "is invalid or it is a duplicate.");
            }
        }

        // Check that these parameters get set
        if (knnMethodContext == null) {
            throw new IllegalArgumentException("Request did not set \"" + KNN_METHOD + "\"");
        }

        if (dimension == -1) {
            throw new IllegalArgumentException("Request did not set \"" + DIMENSION + "\"");
        }

        if (trainingIndex == null) {
            throw new IllegalArgumentException("Request did not set \"" + TRAIN_INDEX_PARAMETER + "\"");
        }

        if (trainingField == null) {
            throw new IllegalArgumentException("Request did not set \"" + TRAIN_FIELD_PARAMETER + "\"");
        }

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(modelId, knnMethodContext, dimension,
                trainingIndex, trainingField, preferredNodeId, description);

        if (maximumVectorCount != -1) {
            trainingModelRequest.setMaximumVectorCount(maximumVectorCount);
        }

        if (searchSize != -1) {
            trainingModelRequest.setSearchSize(searchSize);
        }

        return trainingModelRequest;
    }
}
