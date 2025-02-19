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

import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.support.clustermanager.AcknowledgedRequest;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.indices.ModelMetadata;

import java.io.IOException;

import static org.opensearch.action.ValidateActions.addValidationError;

/**
 * Request for updating model metadata on model system index
 */
public class UpdateModelMetadataRequest extends AcknowledgedRequest<UpdateModelMetadataRequest> {

    private String modelId;
    private boolean isRemoveRequest;
    private ModelMetadata modelMetadata;

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException if read from stream fails
     */
    public UpdateModelMetadataRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readString();
        this.isRemoveRequest = in.readBoolean();

        // modelMetadata cant be null if it is an add request
        if (!this.isRemoveRequest) {
            this.modelMetadata = new ModelMetadata(in);
        }
    }

    /**
     * Constructor
     *
     * @param modelId Id of model
     * @param isRemoveRequest should this model id be removed
     * @param modelMetadata Metadata for model
     */
    public UpdateModelMetadataRequest(String modelId, boolean isRemoveRequest, ModelMetadata modelMetadata) {
        super();
        this.modelId = modelId;
        this.isRemoveRequest = isRemoveRequest;
        this.modelMetadata = modelMetadata;

    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException validationException = null;

        if (modelId.isEmpty()) {
            validationException = addValidationError("Missing model ID", validationException);
        }

        if (!isRemoveRequest && modelMetadata == null) {
            validationException = addValidationError("Model metadata must be passed on add", validationException);
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

    /**
     * Getter for model metadata
     *
     * @return modelMetadata
     */
    public ModelMetadata getModelMetadata() {
        return modelMetadata;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelId);
        out.writeBoolean(isRemoveRequest);

        // Only write metadata if it is an add request
        if (!isRemoveRequest) {
            modelMetadata.writeTo(out);
        }
    }
}
