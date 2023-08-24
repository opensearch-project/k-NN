/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.support.broadcast.BroadcastRequest;
import org.opensearch.core.common.io.stream.StreamInput;

import java.io.IOException;

/**
 * Clear Cache Request. This request contains a list of indices which needs to be evicted from Cache.
 */
public class ClearCacheRequest extends BroadcastRequest<ClearCacheRequest> {

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException if read from stream fails
     */
    public ClearCacheRequest(StreamInput in) throws IOException {
        super(in);
    }

    /**
     * Constructor
     *
     * @param indices list of indices which needs to be evicted from cache
     */
    public ClearCacheRequest(String... indices) {
        super(indices);
    }
}
