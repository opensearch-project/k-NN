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

import org.opensearch.action.support.nodes.BaseNodesRequest;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import java.io.IOException;

/**
 * Request used to ask some or all nodes in the cluster to remove a model from their cache.
 */
public class RemoveModelFromCacheRequest extends BaseNodesRequest<RemoveModelFromCacheRequest> {

    private final String modelId;

    /**
     * Constructor.
     *
     * @param modelId Id of model to be removed from the cache
     * @param nodeIds Id's of nodes clear the model from
     */
    public RemoveModelFromCacheRequest(String modelId, String... nodeIds) {
        super(nodeIds);
        this.modelId = modelId;
    }

    /**
     * Constructor.
     *
     * @param in input stream
     * @throws IOException thrown when reaing input stream fails
     */
    public RemoveModelFromCacheRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelId);
    }

    /**
     * Get model id request
     *
     * @return modelId
     */
    public String getModelId() {
        return modelId;
    }
}
