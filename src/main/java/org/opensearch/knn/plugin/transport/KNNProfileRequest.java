/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.Getter;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.support.broadcast.BroadcastRequest;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;

@Getter
public class KNNProfileRequest extends BroadcastRequest<KNNProfileRequest> {

    private final String index;
    private final String field;

    /**
     * Constructor
     */
    public KNNProfileRequest(String index, String field) {
        super();
        this.index = index;
        this.field = field;
    }

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException in case of I/O errors
     */
    public KNNProfileRequest(StreamInput in) throws IOException {
        super(in);
        index = in.readString();
        field = in.readString();
    }

    public ActionRequestValidationException validate() {
        return null;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(index);
        out.writeString(field);
    }
}
