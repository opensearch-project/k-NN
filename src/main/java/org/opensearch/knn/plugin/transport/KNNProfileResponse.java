/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.opensearch.action.support.broadcast.BroadcastResponse;
import org.opensearch.core.action.support.DefaultShardOperationFailedException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.profiler.SegmentProfilerState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KNNProfileResponse extends BroadcastResponse implements ToXContentObject {

    List<KNNIndexShardProfileResult> shardProfileResults;

    public KNNProfileResponse() {}

    public KNNProfileResponse(StreamInput in) throws IOException {
        super(in);
        int size = in.readInt();
        shardProfileResults = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            shardProfileResults.add(new KNNIndexShardProfileResult(in));
        }
    }

    public KNNProfileResponse(
        List<KNNIndexShardProfileResult> shardProfileResults,
        int totalShards,
        int successfulShards,
        int failedShards,
        List<DefaultShardOperationFailedException> shardFailures
    ) {
        super(totalShards, successfulShards, failedShards, shardFailures);

        this.shardProfileResults = shardProfileResults;
    }

    @Override
    public void writeTo(StreamOutput streamOutput) throws IOException {
        super.writeTo(streamOutput);
        streamOutput.writeInt(shardProfileResults.size());

        for (KNNIndexShardProfileResult result : shardProfileResults) {
            result.writeTo(streamOutput);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();

        builder.field("total_shards", getTotalShards())
            .field("successful_shards", getSuccessfulShards())
            .field("failed_shards", getFailedShards())
            .startObject("shard_profiles");

        // Add shard profile results
        for (KNNIndexShardProfileResult shardProfileResult : shardProfileResults) {
            builder.startObject(shardProfileResult.shardId);

            // Individual segment statistics
            builder.startArray("segments");
            for (SegmentProfilerState state : shardProfileResult.segmentProfilerStateList) {
                builder.startObject()
                    .field("segment_id", state.getSegmentId())
                    .field("dimension", state.getDimension())
                    .startArray("vector_statistics");

                for (int i = 0; i < state.getStatistics().size(); i++) {
                    SummaryStatistics stats = state.getStatistics().get(i);
                    builder.startObject()
                        .field("dimension_index", i)
                        .startObject("statistics")
                        .field("count", stats.getN())
                        .field("min", stats.getMin())
                        .field("max", stats.getMax())
                        .field("sum", stats.getSum())
                        .field("mean", stats.getMean())
                        .field("geometric_mean", stats.getGeometricMean())
                        .field("variance", stats.getVariance())
                        .field("std_deviation", stats.getStandardDeviation())
                        .field("sum_of_squares", stats.getSumsq())
                        .endObject()
                        .endObject();
                }

                builder.endArray().endObject();
            }
            builder.endArray();

            // Aggregated statistics for all segments in this shard
            if (!shardProfileResult.segmentProfilerStateList.isEmpty()) {
                SegmentProfilerState firstState = shardProfileResult.segmentProfilerStateList.get(0);
                int dimensionCount = firstState.getDimension();

                builder.startObject("aggregated")
                    .field("total_segments", shardProfileResult.segmentProfilerStateList.size())
                    .field("dimension", dimensionCount)
                    .startArray("dimensions");

                for (int dim = 0; dim < dimensionCount; dim++) {
                    // Initialize aggregation variables
                    long totalCount = 0;
                    double min = Double.POSITIVE_INFINITY;
                    double max = Double.NEGATIVE_INFINITY;
                    double sumOfValues = 0.0;
                    double sumOfSquares = 0.0;

                    // Combine statistics properly
                    for (SegmentProfilerState state : shardProfileResult.segmentProfilerStateList) {
                        if (dim < state.getStatistics().size()) {
                            SummaryStatistics stats = state.getStatistics().get(dim);

                            totalCount += stats.getN();
                            min = Math.min(min, stats.getMin());
                            max = Math.max(max, stats.getMax());
                            sumOfValues += stats.getSum();
                            sumOfSquares += stats.getSumsq();
                        }
                    }

                    // Calculate aggregate statistics
                    double mean = totalCount > 0 ? sumOfValues / totalCount : 0.0;
                    double variance = totalCount > 0 ? (sumOfSquares / totalCount) - (mean * mean) : 0.0;

                    builder.startObject()
                        .field("dimension_id", dim)
                        .field("count", totalCount)
                        .field("min", min == Double.POSITIVE_INFINITY ? 0.0 : min)
                        .field("max", max == Double.NEGATIVE_INFINITY ? 0.0 : max)
                        .field("mean", mean)
                        .field("std_deviation", Math.sqrt(variance))
                        .field("sum", sumOfValues)
                        .field("sum_of_squares", sumOfSquares)
                        .field("variance", variance)
                        .endObject();
                }
                builder.endArray().endObject();
            }

            builder.endObject();
        }
        builder.endObject();

        // Add any shard failures
        if (getShardFailures() != null && getShardFailures().length > 0) {
            builder.startArray("failures");
            for (DefaultShardOperationFailedException failure : getShardFailures()) {
                failure.toXContent(builder, params);
            }
            builder.endArray();
        }

        return builder.endObject();
    }
}
