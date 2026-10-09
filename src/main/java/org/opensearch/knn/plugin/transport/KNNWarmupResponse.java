/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.ExceptionsHelper;
import org.opensearch.action.support.broadcast.BroadcastResponse;
import org.opensearch.core.action.ShardOperationFailedException;
import org.opensearch.core.action.support.DefaultShardOperationFailedException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.ToXContentObject;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Response returned for k-NN Warmup. Returns total number of shards Warmup was performed on, as well as
 * the number of shards that succeeded and the number of shards that failed.
 * <p>
 * Additionally, reports how many shards were skipped along with the reasons, so that callers can distinguish
 * "warmup completed" from "warmup was a no-op". Skipped shards are a subset of the successful shards.
 * <p>
 * Skip reasons are exposed as a distinct {@code skip_reasons} array rather than a singular value,
 * because a warmup request may target multiple indices whose shards are skipped for different
 * reasons.
 */
public class KNNWarmupResponse extends BroadcastResponse implements ToXContentObject {

    public static final String SHARDS_FIELD = "_shards";
    public static final String TOTAL_FIELD = "total";
    public static final String SUCCESSFUL_FIELD = "successful";
    public static final String SKIPPED_FIELD = "skipped";
    public static final String SKIP_REASONS_FIELD = "skip_reasons";
    public static final String FAILED_FIELD = "failed";
    public static final String FAILURES_FIELD = "failures";

    private int skippedShards;
    private List<String> skipReasons = Collections.emptyList();

    public KNNWarmupResponse() {}

    public KNNWarmupResponse(StreamInput in) throws IOException {
        super(in);
        if (in.getVersion().onOrAfter(KNNWarmupShardResult.MINIMAL_SKIP_STATUS_VERSION)) {
            skippedShards = in.readVInt();
            skipReasons = List.of(in.readStringArray());
        } else {
            skippedShards = 0;
            skipReasons = Collections.emptyList();
        }
    }

    /**
     * Creates a response without skip information (all shards executed).
     */
    public KNNWarmupResponse(
        int totalShards,
        int successfulShards,
        int failedShards,
        List<DefaultShardOperationFailedException> shardFailures
    ) {
        this(totalShards, successfulShards, failedShards, shardFailures, 0, Collections.emptyList());
    }

    /**
     * Creates a response with skip information.
     *
     * @param skippedShards number of shards whose warmup was skipped
     * @param skipReasons distinct reasons the warmups were skipped; empty when nothing was skipped
     */
    public KNNWarmupResponse(
        int totalShards,
        int successfulShards,
        int failedShards,
        List<DefaultShardOperationFailedException> shardFailures,
        int skippedShards,
        List<String> skipReasons
    ) {
        super(totalShards, successfulShards, failedShards, shardFailures);
        this.skippedShards = skippedShards;
        this.skipReasons = List.copyOf(Objects.requireNonNull(skipReasons, "skipReasons must not be null"));
    }

    /**
     * @return number of shards whose warmup was skipped (subset of successful shards)
     */
    public int getSkippedShards() {
        return skippedShards;
    }

    /**
     * @return distinct reasons the warmups were skipped; empty when nothing was skipped
     */
    public List<String> getSkipReasons() {
        return skipReasons;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        if (out.getVersion().onOrAfter(KNNWarmupShardResult.MINIMAL_SKIP_STATUS_VERSION)) {
            out.writeVInt(skippedShards);
            out.writeStringArray(skipReasons.toArray(new String[0]));
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.startObject(SHARDS_FIELD);
        builder.field(TOTAL_FIELD, getTotalShards());
        builder.field(SUCCESSFUL_FIELD, getSuccessfulShards());
        builder.field(SKIPPED_FIELD, skippedShards);
        builder.field(FAILED_FIELD, getFailedShards());
        if (skippedShards > 0) {
            builder.field(SKIP_REASONS_FIELD, skipReasons);
        }
        ShardOperationFailedException[] shardFailures = getShardFailures();
        if (shardFailures != null && shardFailures.length > 0) {
            builder.startArray(FAILURES_FIELD);
            for (ShardOperationFailedException failure : ExceptionsHelper.groupBy(shardFailures)) {
                failure.toXContent(builder, params);
            }
            builder.endArray();
        }
        builder.endObject();
        builder.endObject();
        return builder;
    }
}
