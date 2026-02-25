/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.Getter;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.support.clustermanager.AcknowledgedRequest;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;

import static org.opensearch.action.ValidateActions.addValidationError;

/**
 * Request for updating model graveyard while processing delete model request
 */
public class UpdateModelGraveyardRequest extends AcknowledgedRequest<UpdateModelGraveyardRequest> {

    @Getter
    private final String modelId;
    @Getter
    private final boolean isRemoveRequest;

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException if read from stream fails
     */
    public UpdateModelGraveyardRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readString();
        this.isRemoveRequest = in.readBoolean();
    }

    /**
     * Constructor
     *
     * @param modelId Id of model
     * @param isRemoveRequest should this model id be removed
     */
    public UpdateModelGraveyardRequest(String modelId, boolean isRemoveRequest) {
        super();
        this.modelId = modelId;
        this.isRemoveRequest = isRemoveRequest;
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException validationException = null;

        if (modelId.isEmpty()) {
            validationException = addValidationError("Missing model ID", validationException);
        }

        return validationException;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelId);
        out.writeBoolean(isRemoveRequest);
    }
}
