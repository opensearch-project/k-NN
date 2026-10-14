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
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.knn.index.warmup.WarmupSkipReason;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Response returned for k-NN Warmup. Returns total number of shards Warmup was performed on, as well as
 * the number of shards that succeeded and the number of shards that failed.
 * <p>
 * Additionally, reports how many shards were skipped because the warmup does not apply to them
 * (warm-tier indices, indices without off-heap structures), so that callers can distinguish
 * "warmup completed" from "warmup was a no-op". Skipped shards are a subset of the successful shards.
 * The skip reasons and the k-NN fields that were not loaded are reported in the per-index details.
 */
public class KNNWarmupResponse extends BroadcastResponse implements ToXContentObject {

    public static final String SHARDS_FIELD = "_shards";
    public static final String TOTAL_FIELD = "total";
    public static final String SUCCESSFUL_FIELD = "successful";
    public static final String SKIPPED_FIELD = "skipped";
    public static final String SKIP_REASONS_FIELD = "skip_reasons";
    public static final String SKIPPED_FIELDS_FIELD = "skipped_fields";
    public static final String FAILED_FIELD = "failed";
    public static final String FAILURES_FIELD = "failures";
    public static final String INDICES_FIELD = "indices";

    private int skippedShards;
    private List<IndexWarmupStats> indexStats = Collections.emptyList();

    public KNNWarmupResponse() {}

    public KNNWarmupResponse(StreamInput in) throws IOException {
        super(in);
        if (in.getVersion().onOrAfter(KNNWarmupShardResult.MINIMAL_SKIP_STATUS_VERSION)) {
            skippedShards = in.readVInt();
            indexStats = in.readList(IndexWarmupStats::new);
        } else {
            skippedShards = 0;
            indexStats = Collections.emptyList();
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
     * Creates a response with skip information and per-index details.
     *
     * @param skippedShards number of shards whose warmup was skipped
     * @param indexStats per-index warmup statistics
     */
    public KNNWarmupResponse(
        int totalShards,
        int successfulShards,
        int failedShards,
        List<DefaultShardOperationFailedException> shardFailures,
        int skippedShards,
        List<IndexWarmupStats> indexStats
    ) {
        super(totalShards, successfulShards, failedShards, shardFailures);
        this.skippedShards = skippedShards;
        this.indexStats = List.copyOf(Objects.requireNonNull(indexStats, "indexStats must not be null"));
    }

    /**
     * @return number of shards whose warmup was skipped (subset of successful shards)
     */
    public int getSkippedShards() {
        return skippedShards;
    }

    /**
     * @return per-index warmup statistics
     */
    public List<IndexWarmupStats> getIndexStats() {
        return indexStats;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        if (out.getVersion().onOrAfter(KNNWarmupShardResult.MINIMAL_SKIP_STATUS_VERSION)) {
            out.writeVInt(skippedShards);
            out.writeList(indexStats);
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
        writeShardFailures(builder, params, getShardFailures());
        builder.endObject();

        if (indexStats.isEmpty() == false) {
            builder.startObject(INDICES_FIELD);
            for (final IndexWarmupStats indexStat : indexStats) {
                builder.startObject(indexStat.getIndexName());
                builder.startObject(SHARDS_FIELD);
                builder.field(TOTAL_FIELD, indexStat.getTotalShards());
                builder.field(SUCCESSFUL_FIELD, indexStat.getSuccessfulShards());
                builder.field(SKIPPED_FIELD, indexStat.getSkippedShards());
                builder.field(FAILED_FIELD, indexStat.getFailedShards());
                builder.endObject();
                if (indexStat.getSkipReasons().isEmpty() == false) {
                    builder.startArray(SKIP_REASONS_FIELD);
                    for (final WarmupSkipReason skipReason : indexStat.getSkipReasons()) {
                        builder.value(skipReason.getValue());
                    }
                    builder.endArray();
                }
                if (indexStat.getSkippedFields().isEmpty() == false) {
                    builder.field(SKIPPED_FIELDS_FIELD, indexStat.getSkippedFields());
                }
                builder.endObject();
            }
            builder.endObject();
        }

        builder.endObject();
        return builder;
    }

    private static void writeShardFailures(
        final XContentBuilder builder,
        final Params params,
        final ShardOperationFailedException[] shardFailures
    ) throws IOException {
        if (shardFailures != null && shardFailures.length > 0) {
            builder.startArray(FAILURES_FIELD);
            for (ShardOperationFailedException failure : ExceptionsHelper.groupBy(shardFailures)) {
                failure.toXContent(builder, params);
            }
            builder.endArray();
        }
    }

    /**
     * Warmup statistics for a single index, aggregated from the shard results of that index.
     */
    public static class IndexWarmupStats implements Writeable {

        private final String indexName;
        private final int totalShards;
        private final int successfulShards;
        private final int skippedShards;
        private final int failedShards;
        private final List<WarmupSkipReason> skipReasons;
        private final List<String> skippedFields;

        public IndexWarmupStats(
            final String indexName,
            final int totalShards,
            final int successfulShards,
            final int skippedShards,
            final int failedShards,
            final List<WarmupSkipReason> skipReasons,
            final List<String> skippedFields
        ) {
            this.indexName = Objects.requireNonNull(indexName, "indexName must not be null");
            this.totalShards = totalShards;
            this.successfulShards = successfulShards;
            this.skippedShards = skippedShards;
            this.failedShards = failedShards;
            this.skipReasons = List.copyOf(Objects.requireNonNull(skipReasons, "skipReasons must not be null"));
            this.skippedFields = List.copyOf(Objects.requireNonNull(skippedFields, "skippedFields must not be null"));
        }

        public IndexWarmupStats(final StreamInput in) throws IOException {
            this.indexName = in.readString();
            this.totalShards = in.readVInt();
            this.successfulShards = in.readVInt();
            this.skippedShards = in.readVInt();
            this.failedShards = in.readVInt();
            this.skipReasons = in.readStringList().stream().map(WarmupSkipReason::fromName).toList();
            this.skippedFields = in.readStringList();
        }

        @Override
        public void writeTo(final StreamOutput out) throws IOException {
            out.writeString(indexName);
            out.writeVInt(totalShards);
            out.writeVInt(successfulShards);
            out.writeVInt(skippedShards);
            out.writeVInt(failedShards);
            out.writeStringCollection(skipReasons.stream().map(WarmupSkipReason::name).toList());
            out.writeStringCollection(skippedFields);
        }

        public String getIndexName() {
            return indexName;
        }

        /**
         * @return number of shards of the index targeted by the warmup, including shards that were
         *         not available (e.g. unassigned replicas)
         */
        public int getTotalShards() {
            return totalShards;
        }

        public int getSuccessfulShards() {
            return successfulShards;
        }

        public int getSkippedShards() {
            return skippedShards;
        }

        public int getFailedShards() {
            return failedShards;
        }

        /**
         * @return distinct reasons the index's shards were skipped; empty when no shard was skipped
         */
        public List<WarmupSkipReason> getSkipReasons() {
            return skipReasons;
        }

        /**
         * @return names of the k-NN fields that were not loaded; empty when all fields were loaded
         */
        public List<String> getSkippedFields() {
            return skippedFields;
        }

        @Override
        public boolean equals(final Object other) {
            if (this == other) {
                return true;
            }
            if (other instanceof IndexWarmupStats == false) {
                return false;
            }
            final IndexWarmupStats that = (IndexWarmupStats) other;
            return totalShards == that.totalShards
                && successfulShards == that.successfulShards
                && skippedShards == that.skippedShards
                && failedShards == that.failedShards
                && Objects.equals(indexName, that.indexName)
                && Objects.equals(skipReasons, that.skipReasons)
                && Objects.equals(skippedFields, that.skippedFields);
        }

        @Override
        public int hashCode() {
            return Objects.hash(indexName, totalShards, successfulShards, skippedShards, failedShards, skipReasons, skippedFields);
        }

        /**
         * Mutable builder, used to aggregate the shard results of one index during response
         * construction.
         */
        static class Builder {

            private final String indexName;
            private int totalShards;
            private int successfulShards;
            private int skippedShards;
            private int failedShards;
            private final Set<WarmupSkipReason> skipReasons = new LinkedHashSet<>();
            private final Set<String> skippedFields = new LinkedHashSet<>();

            Builder(final String indexName) {
                this.indexName = indexName;
            }

            void targetedShard() {
                totalShards++;
            }

            void successfulShard(final KNNWarmupShardResult shardResult) {
                successfulShards++;
                if (shardResult.isSkipped()) {
                    skippedShards++;
                    skipReasons.add(shardResult.getSkipReason());
                }
                skippedFields.addAll(shardResult.getSkippedFields());
            }

            void failedShard() {
                failedShards++;
            }

            IndexWarmupStats build() {
                return new IndexWarmupStats(
                    indexName,
                    Math.max(totalShards, successfulShards + failedShards),
                    successfulShards,
                    skippedShards,
                    failedShards,
                    List.copyOf(skipReasons),
                    List.copyOf(skippedFields)
                );
            }
        }
    }

    /**
     * Accumulates per-index statistics from the targeted shards, shard results and shard failures
     * during response aggregation.
     */
    static class IndexStatsAccumulator {

        private final Map<String, IndexWarmupStats.Builder> byIndex = new LinkedHashMap<>();

        /**
         * Registers a shard targeted by the warmup. Shards that were not available are only known
         * from the routing table, so the per-index total is counted here rather than from results.
         */
        void addTargetedShard(final String indexName) {
            builder(indexName).targetedShard();
        }

        void add(final KNNWarmupShardResult shardResult) {
            // Results from pre-3.10.0 nodes carry no index name and cannot be attributed.
            if (shardResult.getIndexName() == null) {
                return;
            }
            builder(shardResult.getIndexName()).successfulShard(shardResult);
        }

        void add(final DefaultShardOperationFailedException shardFailure) {
            // Node level failures may not carry an index name.
            if (shardFailure.index() == null) {
                return;
            }
            builder(shardFailure.index()).failedShard();
        }

        List<IndexWarmupStats> build() {
            final List<IndexWarmupStats> stats = new ArrayList<>(byIndex.size());
            for (final IndexWarmupStats.Builder builder : byIndex.values()) {
                stats.add(builder.build());
            }
            return stats;
        }

        private IndexWarmupStats.Builder builder(final String indexName) {
            return byIndex.computeIfAbsent(indexName, IndexWarmupStats.Builder::new);
        }
    }
}
