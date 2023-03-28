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
import org.opensearch.common.Nullable;
import org.opensearch.common.Strings;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
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
    private final String modelID;
    private final String result;
    private final String errorMessage;

    public DeleteModelResponse(String modelID, String result, @Nullable String errorMessage) {
        this.modelID = modelID;
        this.result = result;
        this.errorMessage = errorMessage;
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
                "result": "not_found",
                "error": "Model my_model_id doesn't exist"
        }
         */
        builder.startObject();
        builder.field(MODEL_ID, getModelID());
        builder.field(RESULT, getResult());
        if (Strings.hasText(errorMessage)) {
            builder.field(ERROR_MSG, getErrorMessage());
        }
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
