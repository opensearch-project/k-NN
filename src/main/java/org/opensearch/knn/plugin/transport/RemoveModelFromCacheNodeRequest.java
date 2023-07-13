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

import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.transport.TransportRequest;

import java.io.IOException;

/**
 * Request sent to each to tell it to evict a given model from the cache.
 */
public class RemoveModelFromCacheNodeRequest extends TransportRequest {

    private final String modelId;

    /**
     * Constructor.
     *
     * @param modelId identifier of the model.
     */
    public RemoveModelFromCacheNodeRequest(String modelId) {
        super();
        this.modelId = modelId;
    }

    /**
     * Constructor from stream
     *
     * @param in input stream
     * @throws IOException thrown when reading from stream fails
     */
    public RemoveModelFromCacheNodeRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelId);
    }

    /**
     * Getter for model id
     *
     * @return modelId
     */
    public String getModelId() {
        return modelId;
    }
}
