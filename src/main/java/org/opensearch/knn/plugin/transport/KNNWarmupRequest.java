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

import org.opensearch.action.support.broadcast.BroadcastRequest;
import org.opensearch.common.io.stream.StreamInput;

import java.io.IOException;

/**
 * k-NN Warmup Request. This request contains a list of indices for which warmup should be performed.
 */
public class KNNWarmupRequest extends BroadcastRequest<KNNWarmupRequest> {

    public KNNWarmupRequest(StreamInput in) throws IOException {
        super(in);
    }

    public KNNWarmupRequest(String... indices) {
        super(indices);
    }
}
