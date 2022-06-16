/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.support.master.AcknowledgedRequest;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;

import static org.opensearch.action.ValidateActions.addValidationError;

/**
 * Request for updating blocked modelIds list while processing delete model request
 */
public class UpdateBlockedModelRequest extends AcknowledgedRequest<UpdateBlockedModelRequest> {

    private String modelId;
    private boolean isRemoveRequest;

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException if read from stream fails
     */
    public UpdateBlockedModelRequest(StreamInput in) throws IOException {
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
    public UpdateBlockedModelRequest(String modelId, boolean isRemoveRequest) {
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

    /**
     * Getter for modelId
     *
     * @return modelId
     */
    public String getModelId() {
        return modelId;
    }

    /**
     * Getter for isRemoveRequest
     *
     * @return isRemoveRequest
     */
    public boolean isRemoveRequest() {
        return isRemoveRequest;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelId);
        out.writeBoolean(isRemoveRequest);
    }
}
