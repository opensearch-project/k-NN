/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.support.broadcast.BroadcastRequest;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;

/**
 * Request for KNN profile operation
 */
public class KNNProfileRequest extends BroadcastRequest<KNNProfileRequest> {
    private String fieldName;

    /**
     * Constructor
     */
    public KNNProfileRequest() {
        super();
    }

    /**
     * Constructor with indices
     *
     * @param indices Indices to profile
     */
    public KNNProfileRequest(String... indices) {
        super(indices);
    }

    /**
     * Constructor from StreamInput
     *
     * @param in StreamInput
     * @throws IOException if there's an error reading from stream
     */
    public KNNProfileRequest(StreamInput in) throws IOException {
        super(in);
        this.fieldName = in.readOptionalString();
    }

    /**
     * Write to StreamOutput
     *
     * @param out StreamOutput
     * @throws IOException if there's an error writing to stream
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeOptionalString(fieldName);
    }

    /**
     * Get field name to profile
     *
     * @return field name
     */
    public String getFieldName() {
        return fieldName;
    }

    /**
     * Set field name to profile
     *
     * @param fieldName field name
     */
    public void setFieldName(String fieldName) {
        this.fieldName = fieldName;
    }
}
