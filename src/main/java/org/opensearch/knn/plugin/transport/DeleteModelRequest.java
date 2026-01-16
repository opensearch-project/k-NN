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

import org.apache.commons.lang3.StringUtils;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;

import static org.opensearch.action.ValidateActions.addValidationError;

public class DeleteModelRequest extends ActionRequest {

    private String modelID;

    public DeleteModelRequest(StreamInput in) throws IOException {
        super(in);
        this.modelID = in.readString();
    }

    public DeleteModelRequest(String modelID) {
        super();
        this.modelID = modelID;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelID);
    }

    @Override
    public ActionRequestValidationException validate() {
        if (StringUtils.isNotBlank(modelID)) {
            return null;
        }
        return addValidationError("Model id cannot be empty ", null);
    }

    public String getModelID() {
        return modelID;
    }
}
