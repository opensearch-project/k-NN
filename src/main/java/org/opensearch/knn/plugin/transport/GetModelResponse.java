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
package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionResponse;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;

import java.io.IOException;
import java.util.Base64;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODEL_ERROR;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODEL_STATE;
import static org.opensearch.knn.common.KNNConstants.MODEL_TIMESTAMP;

/**
 * {@link GetModelResponse} represents Response returned by {@link GetModelRequest}
 */
public class GetModelResponse extends ActionResponse implements ToXContentObject {

    private final String modelID;
    private final Model model;

    public GetModelResponse(String modelID, Model model) {
        this.modelID = modelID;
        this.model = model;
    }

    public GetModelResponse(StreamInput in) throws IOException {
        super(in);
        this.modelID = in.readString();
        ModelMetadata metadata = new ModelMetadata(in);
        this.model = new Model(metadata, in.readByteArray());
    }

    public String getModelID() {
        return modelID;
    }

    public Model getModel() {
        return model;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        /* Response should look like below:
            {
                "model_id": "my_model_id"
                "state": "created",
                "created_timestamp": "10-31-21 02:02:02",
                "description": "Model trained with dataset X",
                "error": "",
                "model_blob": "cdscsacsadcsdca",
                "engine": "faiss",
                "space_type": "l2",
                "dimension": 128
        }
         */
        builder.startObject();
        builder.field(MODEL_ID, modelID);
        builder.field(MODEL_STATE, model.getModelMetadata().getState().getName());
        builder.field(MODEL_TIMESTAMP, model.getModelMetadata().getTimestamp());
        builder.field(MODEL_DESCRIPTION, model.getModelMetadata().getDescription());
        builder.field(MODEL_ERROR, model.getModelMetadata().getError());

        String base64Model = Base64.getEncoder().encodeToString(model.getModelBlob());
        builder.field(MODEL_BLOB_PARAMETER, base64Model);

        builder.field(METHOD_PARAMETER_SPACE_TYPE, model.getModelMetadata().getSpaceType().getValue());
        builder.field(DIMENSION, model.getModelMetadata().getDimension());
        builder.field(KNN_ENGINE, model.getModelMetadata().getKnnEngine().getName());

        builder.endObject();

        return builder;
    }

    @Override
    public void writeTo(StreamOutput output) throws IOException {
        output.writeString(modelID);
        model.getModelMetadata().writeTo(output);
        output.writeByteArray(model.getModelBlob());
    }
}
