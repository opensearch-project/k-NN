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

import org.opensearch.common.Nullable;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

/**
 * {@link DeleteModelResponse} represents Response returned by {@link DeleteModelRequest}
 */
public class DeleteModelResponse extends ActionResponse implements ToXContentObject {

    public static final String RESULT = "result";
    public static final String ERROR_MSG = "error";
    private static final String DELETED = "deleted";
    private final String modelID;
    private final String result;
    private final String errorMessage;

    /**
     * Ctor to build delete model response.
     * @deprecated
     * Returning errors through {@link DeleteModelResponse} should not be done. Instead, if there is an
     * error, throw/return a suitable exception. Use {@link DeleteModelResponse#DeleteModelResponse(String)} to
     * construct valid responses instead.
     *
     * @param modelID ID of the model that is deleted
     * @param result Resulting action of the deletion.
     * @param errorMessage Error message to be returned to the user
     */
    @Deprecated
    public DeleteModelResponse(String modelID, String result, @Nullable String errorMessage) {
        this.modelID = modelID;
        this.result = result;
        this.errorMessage = errorMessage;
    }

    /**
     * Ctor to build delete model response
     *
     * @param modelID ID of the model that is deleted
     */
    public DeleteModelResponse(String modelID) {
        this.modelID = modelID;
        this.result = DELETED;
        this.errorMessage = null;
    }

    public DeleteModelResponse(StreamInput in) throws IOException {
        super(in);
        this.modelID = in.readString();
        this.result = in.readString();
        this.errorMessage = in.readOptionalString();
    }

    public String getModelID() {
        return modelID;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public String getResult() {
        return result;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        /* Response should look like below:
            {
                "model_id": "my_model_id"
                "result": "deleted"
        }
         */
        builder.startObject();
        builder.field(MODEL_ID, getModelID());
        builder.field(RESULT, getResult());
        builder.endObject();
        return builder;

    }

    @Override
    public void writeTo(StreamOutput output) throws IOException {
        output.writeString(modelID);
        output.writeString(getResult());
        output.writeOptionalString(getErrorMessage());
    }
}
