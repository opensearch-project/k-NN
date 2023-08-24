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

import org.opensearch.core.action.ActionResponse;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;

/**
 * Response for training model request
 */
public class TrainingModelResponse extends ActionResponse implements ToXContentObject {

    private String modelId;

    /**
     * Constructor.
     *
     * @param modelId of model to be trained
     */
    public TrainingModelResponse(String modelId) {
        this.modelId = modelId;
    }

    /**
     * Constructor from stream.
     *
     * @param in StreamInput to read from
     * @throws IOException on failure to read from stream
     */
    public TrainingModelResponse(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readOptionalString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeOptionalString(this.modelId);
    }

    /**
     * Getter for modelId
     *
     * @return modelId that was created via train request
     */
    public String getModelId() {
        return modelId;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(KNNConstants.MODEL_ID, this.modelId);
        builder.endObject();
        return builder;
    }
}
