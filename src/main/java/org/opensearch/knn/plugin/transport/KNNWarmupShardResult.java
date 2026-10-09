/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.Version;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.knn.index.warmup.WarmupSkipReason;

import java.io.IOException;
import java.util.Objects;

/**
 * Result of a warmup operation on a single shard. Carries whether the warmup was actually performed
 * or skipped, and when skipped, the reason why.
 */
public class KNNWarmupShardResult implements Writeable {

    /**
     * Version as of which the skip status fields are written to / read from the wire.
     */
    public static final Version MINIMAL_SKIP_STATUS_VERSION = Version.V_3_9_0;

    /**
     * Shared instance for shards whose warmup was executed (not skipped).
     */
    private static final KNNWarmupShardResult EXECUTED = new KNNWarmupShardResult(false, null);

    private final boolean skipped;
    private final WarmupSkipReason skipReason;

    /**
     * Creates a result for a shard whose warmup was executed.
     */
    public static KNNWarmupShardResult executed() {
        return EXECUTED;
    }

    /**
     * Creates a result for a shard whose warmup was skipped.
     *
     * @param skipReason reason the warmup was skipped, must not be null
     * @return skipped result
     */
    public static KNNWarmupShardResult skipped(final WarmupSkipReason skipReason) {
        return new KNNWarmupShardResult(true, Objects.requireNonNull(skipReason, "skipReason must not be null"));
    }

    private KNNWarmupShardResult(final boolean skipped, final WarmupSkipReason skipReason) {
        this.skipped = skipped;
        this.skipReason = skipReason;
    }

    public KNNWarmupShardResult(final StreamInput in) throws IOException {
        if (in.getVersion().onOrAfter(MINIMAL_SKIP_STATUS_VERSION)) {
            this.skipped = in.readBoolean();
            this.skipReason = this.skipped ? resolveSkipReason(in.readString()) : null;
        } else {
            this.skipped = false;
            this.skipReason = null;
        }
    }

    private static WarmupSkipReason resolveSkipReason(final String name) {
        try {
            return WarmupSkipReason.valueOf(name);
        } catch (IllegalArgumentException e) {
            return WarmupSkipReason.UNKNOWN;
        }
    }

    @Override
    public void writeTo(final StreamOutput out) throws IOException {
        if (out.getVersion().onOrAfter(MINIMAL_SKIP_STATUS_VERSION)) {
            out.writeBoolean(skipped);
            if (skipped) {
                out.writeString(skipReason.name());
            }
        }
        // Before 3.9.0 the shard result was an empty payload; keep the wire format identical.
    }

    /**
     * @return true if the warmup was skipped for this shard, false if it was executed
     */
    public boolean isSkipped() {
        return skipped;
    }

    /**
     * @return the reason the warmup was skipped; null when the warmup was executed
     */
    public WarmupSkipReason getSkipReason() {
        return skipReason;
    }
}
