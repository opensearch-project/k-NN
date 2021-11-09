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

import org.opensearch.action.support.DefaultShardOperationFailedException;
import org.opensearch.action.support.broadcast.BroadcastResponse;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.xcontent.ToXContentObject;

import java.io.IOException;
import java.util.List;

/**
 * Response returned for k-NN Warmup. Returns total number of shards Warmup was performed on, as well as
 * the number of shards that succeeded and the number of shards that failed.
 */
public class KNNWarmupResponse extends BroadcastResponse implements ToXContentObject {

    public KNNWarmupResponse() {}

    public KNNWarmupResponse(StreamInput in) throws IOException {
        super(in);
    }

    public KNNWarmupResponse(int totalShards, int successfulShards, int failedShards,
                             List<DefaultShardOperationFailedException> shardFailures) {
        super(totalShards, successfulShards, failedShards, shardFailures);
    }
}
