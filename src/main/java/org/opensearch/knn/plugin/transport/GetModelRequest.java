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

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;

/**
 * {@link GetModelRequest} gets Model for given modelID
 */
public class GetModelRequest extends ActionRequest {

    private String modelID;

    /**
     * Constructor
     *
     * @param modelID model ID of Index Model
     */
    public GetModelRequest(String modelID) {
        super();
        this.modelID = modelID;
    }

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException in case of I/O errors
     */
    public GetModelRequest(StreamInput in) throws IOException {
        super(in);
        modelID = in.readString();
    }

    @Override
    public ActionRequestValidationException validate() {
        return null;
    }

    public String getModelID() {
        return modelID;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelID);
    }
}
