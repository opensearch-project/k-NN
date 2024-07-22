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
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.indices.ModelUtil;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.TrainingJobRouterAction;
import org.opensearch.knn.plugin.transport.TrainingModelRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.MAX_VECTOR_COUNT_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PREFERENCE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SEARCH_SIZE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Rest Handler for model training api endpoint.
 */
public class RestTrainModelHandler extends BaseRestHandler {

    private final static String NAME = "knn_train_model_action";
    private final static Object DEFAULT_NOT_SET_OBJECT_VALUE = null;
    private final static int DEFAULT_NOT_SET_INT_VALUE = -1;

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList.of(
            new Route(RestRequest.Method.POST, String.format(Locale.ROOT, "%s/%s/{%s}/_train", KNNPlugin.KNN_BASE_URI, MODELS, MODEL_ID)),
            new Route(RestRequest.Method.POST, String.format(Locale.ROOT, "%s/%s/_train", KNNPlugin.KNN_BASE_URI, MODELS))
        );
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) throws IOException {
        TrainingModelRequest trainingModelRequest = createTransportRequest(restRequest);

        return channel -> client.execute(TrainingJobRouterAction.INSTANCE, trainingModelRequest, new RestToXContentListener<>(channel));
    }

    private TrainingModelRequest createTransportRequest(RestRequest restRequest) throws IOException {
        // Parse query params
        String modelId = restRequest.param(MODEL_ID);
        String preferredNodeId = restRequest.param(PREFERENCE_PARAMETER);

        // Parse request body
        XContentParser parser = restRequest.contentParser();
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);

        KNNMethodContext knnMethodContext = (KNNMethodContext) DEFAULT_NOT_SET_OBJECT_VALUE;
        String trainingIndex = (String) DEFAULT_NOT_SET_OBJECT_VALUE;
        String trainingField = (String) DEFAULT_NOT_SET_OBJECT_VALUE;
        String description = (String) DEFAULT_NOT_SET_OBJECT_VALUE;
        VectorDataType vectorDataType = (VectorDataType) DEFAULT_NOT_SET_OBJECT_VALUE;

        int dimension = DEFAULT_NOT_SET_INT_VALUE;
        int maximumVectorCount = DEFAULT_NOT_SET_INT_VALUE;
        int searchSize = DEFAULT_NOT_SET_INT_VALUE;

        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            if (TRAIN_INDEX_PARAMETER.equals(fieldName) && ensureNotSet(fieldName, trainingIndex)) {
                trainingIndex = parser.textOrNull();
            } else if (TRAIN_FIELD_PARAMETER.equals(fieldName) && ensureNotSet(fieldName, trainingField)) {
                trainingField = parser.textOrNull();
            } else if (KNN_METHOD.equals(fieldName) && ensureNotSet(fieldName, knnMethodContext)) {
                knnMethodContext = KNNMethodContext.parse(parser.map());
                if (SpaceType.UNDEFINED == knnMethodContext.getSpaceType()) {
                    knnMethodContext.setSpaceType(SpaceType.L2);
                }
            } else if (DIMENSION.equals(fieldName) && ensureNotSet(fieldName, dimension)) {
                dimension = (Integer) NumberFieldMapper.NumberType.INTEGER.parse(parser.objectBytes(), false);
            } else if (MAX_VECTOR_COUNT_PARAMETER.equals(fieldName) && ensureNotSet(fieldName, maximumVectorCount)) {
                maximumVectorCount = (Integer) NumberFieldMapper.NumberType.INTEGER.parse(parser.objectBytes(), false);
            } else if (SEARCH_SIZE_PARAMETER.equals(fieldName) && ensureNotSet(fieldName, searchSize)) {
                searchSize = (Integer) NumberFieldMapper.NumberType.INTEGER.parse(parser.objectBytes(), false);
            } else if (MODEL_DESCRIPTION.equals(fieldName) && ensureNotSet(fieldName, description)) {
                description = parser.textOrNull();
                ModelUtil.blockCommasInModelDescription(description);
            } else if (VECTOR_DATA_TYPE_FIELD.equals(fieldName) && ensureNotSet(fieldName, vectorDataType)) {
                vectorDataType = VectorDataType.get(parser.text());
            } else {
                throw new IllegalArgumentException("Unable to parse token. \"" + fieldName + "\" is not a valid " + "parameter.");
            }
        }

        // Check that these parameters get set
        ensureSet(KNN_METHOD, knnMethodContext);
        ensureSet(DIMENSION, dimension);
        ensureSet(TRAIN_INDEX_PARAMETER, trainingIndex);
        ensureSet(TRAIN_FIELD_PARAMETER, trainingField);

        // Convert null description to empty string.
        if (description == DEFAULT_NOT_SET_OBJECT_VALUE) {
            description = "";
        }

        if (vectorDataType == DEFAULT_NOT_SET_OBJECT_VALUE) {
            vectorDataType = VectorDataType.DEFAULT;
        }

        TrainingModelRequest trainingModelRequest = new TrainingModelRequest(
            modelId,
            knnMethodContext,
            dimension,
            trainingIndex,
            trainingField,
            preferredNodeId,
            description,
            vectorDataType
        );

        if (maximumVectorCount != DEFAULT_NOT_SET_INT_VALUE) {
            trainingModelRequest.setMaximumVectorCount(maximumVectorCount);
        }

        if (searchSize != DEFAULT_NOT_SET_INT_VALUE) {
            trainingModelRequest.setSearchSize(searchSize);
        }

        return trainingModelRequest;
    }

    private void ensureSet(String fieldName, Object value) {
        if (value == DEFAULT_NOT_SET_OBJECT_VALUE) {
            throw new IllegalArgumentException("Request did not set \"" + fieldName + ".");
        }
    }

    private void ensureSet(String fieldName, int value) {
        if (value == DEFAULT_NOT_SET_INT_VALUE) {
            throw new IllegalArgumentException("Request did not set \"" + fieldName + ".");
        }
    }

    private boolean ensureNotSet(String fieldName, Object value) {
        if (value != DEFAULT_NOT_SET_OBJECT_VALUE) {
            throw new IllegalArgumentException("Unable to parse token. \"" + fieldName + "\" is duplicated.");
        }

        return true;
    }

    private boolean ensureNotSet(String fieldName, int value) {
        if (value != DEFAULT_NOT_SET_INT_VALUE) {
            throw new IllegalArgumentException("Unable to parse token. \"" + fieldName + "\" is duplicated.");
        }

        return true;
    }
}
