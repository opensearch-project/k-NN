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
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Result of a warmup operation on a single shard. Carries the name of the index the shard belongs
 * to, whether the warmup was actually performed or skipped (and when skipped, the reason why),
 * and the names of the k-NN fields that were not loaded. Aggregated by
 * {@link KNNWarmupTransportAction#newResponse} into {@link KNNWarmupResponse}.
 * <p>
 * Wire format compatibility: prior to 3.10.0 the shard result carried no information at all
 * ({@link org.opensearch.action.support.broadcast.node.TransportBroadcastByNodeAction.EmptyResult}
 * writes zero bytes). To stay byte-compatible during rolling upgrades, this result writes nothing
 * when the stream version is before 3.10.0, and readers default to "executed, no index, no fields".
 */
public class KNNWarmupShardResult implements Writeable {

    /**
     * Version as of which the skip status fields are written to / read from the wire.
     */
    public static final Version MINIMAL_SKIP_STATUS_VERSION = Version.V_3_10_0;

    private final String indexName;
    private final boolean skipped;
    private final WarmupSkipReason skipReason;
    private final List<String> skippedFields;

    /**
     * Creates a result for a shard whose warmup was executed.
     *
     * @param indexName name of the index the shard belongs to
     * @param skippedFields names of the k-NN fields that were not loaded; may be empty
     */
    public static KNNWarmupShardResult executed(final String indexName, final List<String> skippedFields) {
        return new KNNWarmupShardResult(indexName, false, null, skippedFields);
    }

    /**
     * Creates a result for a shard whose warmup was skipped.
     *
     * @param indexName name of the index the shard belongs to
     * @param skipReason reason the warmup was skipped, must not be null
     * @param skippedFields names of the k-NN fields that were not loaded; may be empty when the
     *        skip is unrelated to fields (e.g. a warm-tier index)
     */
    public static KNNWarmupShardResult skipped(
        final String indexName,
        final WarmupSkipReason skipReason,
        final List<String> skippedFields
    ) {
        return new KNNWarmupShardResult(indexName, true, Objects.requireNonNull(skipReason, "skipReason must not be null"), skippedFields);
    }

    private KNNWarmupShardResult(
        final String indexName,
        final boolean skipped,
        final WarmupSkipReason skipReason,
        final List<String> skippedFields
    ) {
        this.indexName = indexName;
        this.skipped = skipped;
        this.skipReason = skipReason;
        this.skippedFields = List.copyOf(Objects.requireNonNull(skippedFields, "skippedFields must not be null"));
    }

    public KNNWarmupShardResult(final StreamInput in) throws IOException {
        if (in.getVersion().onOrAfter(MINIMAL_SKIP_STATUS_VERSION)) {
            this.skipped = in.readBoolean();
            this.skipReason = this.skipped ? WarmupSkipReason.fromName(in.readString()) : null;
            this.indexName = in.readString();
            this.skippedFields = in.readStringList();
        } else {
            this.skipped = false;
            this.skipReason = null;
            this.indexName = null;
            this.skippedFields = Collections.emptyList();
        }
    }

    @Override
    public void writeTo(final StreamOutput out) throws IOException {
        if (out.getVersion().onOrAfter(MINIMAL_SKIP_STATUS_VERSION)) {
            out.writeBoolean(skipped);
            if (skipped) {
                out.writeString(skipReason.name());
            }
            out.writeString(indexName);
            out.writeStringCollection(skippedFields);
        }
    }

    /**
     * @return name of the index the shard belongs to; null for results deserialized from pre-3.10.0
     *         nodes, which cannot be attributed to an index
     */
    public String getIndexName() {
        return indexName;
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

    /**
     * @return names of the k-NN fields that were not loaded; empty when all fields were loaded
     */
    public List<String> getSkippedFields() {
        return skippedFields;
    }
}
