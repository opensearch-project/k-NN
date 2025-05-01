/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues;
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

/**
 * Response object for KNN profile requests that provides statistical information about vectors
 * at segment, shard, and cluster levels.
 *
 * Example response:
 * {
 *   "total_shards": 2,
 *   "successful_shards": 2,
 *   "failed_shards": 0,
 *   "shard_profiles": {
 *     "0": {
 *       "segments": [
 *         {
 *           "segment_id": "_0",
 *           "dimension": 128,
 *           "vector_statistics": [
 *             {
 *               "dimension_index": 0,
 *               "statistics": {
 *                 "count": 1000,
 *                 "min": -0.523,
 *                 "max": 0.785,
 *                 "sum": 156.78,
 *                 "mean": 0.157,
 *                 "geometric_mean": 0.145,
 *                 "variance": 0.089,
 *                 "std_deviation": 0.298,
 *                 "sum_of_squares": 245.67
 *               }
 *             }
 *             // ... additional dimensions
 *           ]
 *         }
 *         // ... additional segments
 *       ],
 *       "aggregated": {
 *         "total_segments": 3,
 *         "dimension": 128,
 *         "dimensions": [
 *           {
 *             "dimension_id": 0,
 *             "count": 3000,
 *             "min": -0.723,
 *             "max": 0.892,
 *             "mean": 0.167,
 *             "std_deviation": 0.312,
 *             "sum": 501.23,
 *             "variance": 0.097
 *           }
 *           // ... additional dimensions
 *         ]
 *       }
 *     }
 *     // ... additional shards
 *   },
 *   "cluster_aggregation": {
 *     "total_shards": 2,
 *     "dimension": 128,
 *     "dimensions": [
 *       {
 *         "dimension_id": 0,
 *         "count": 6000,
 *         "min": -0.723,
 *         "max": 0.892,
 *         "mean": 0.172,
 *         "std_deviation": 0.315,
 *         "sum": 1032.45,
 *         "variance": 0.099
 *       }
 *       // ... additional dimensions
 *     ]
 *   },
 *   "failures": []
 * }
 */
public class KNNProfileResponse extends BroadcastResponse implements ToXContentObject {

    List<KNNIndexShardProfileResult> shardProfileResults;

    public KNNProfileResponse() {}

    public KNNProfileResponse(StreamInput in) throws IOException {
        super(in);
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
        throw new UnsupportedOperationException("This method is not available");
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
                    List<StatisticalSummary> dimensionStats = new ArrayList<>();

                    // Collect statistics from all segments for this dimension
                    for (SegmentProfilerState state : shardProfileResult.segmentProfilerStateList) {
                        if (dim < state.getStatistics().size()) {
                            dimensionStats.add(state.getStatistics().get(dim));
                        }
                    }

                    // Use AggregateSummaryStatistics to combine segment statistics
                    StatisticalSummaryValues aggregatedStats = AggregateSummaryStatistics.aggregate(dimensionStats);

                    builder.startObject()
                        .field("dimension_id", dim)
                        .field("count", aggregatedStats.getN())
                        .field("min", aggregatedStats.getMin())
                        .field("max", aggregatedStats.getMax())
                        .field("mean", aggregatedStats.getMean())
                        .field("std_deviation", Math.sqrt(aggregatedStats.getVariance()))
                        .field("sum", aggregatedStats.getSum())
                        .field("variance", aggregatedStats.getVariance())
                        .endObject();
                }
                builder.endArray().endObject();
            }

            builder.endObject();
        }
        builder.endObject();

        // Add cluster-level aggregation
        if (!shardProfileResults.isEmpty() && !shardProfileResults.get(0).segmentProfilerStateList.isEmpty()) {
            SegmentProfilerState firstState = shardProfileResults.get(0).segmentProfilerStateList.get(0);
            int dimensionCount = firstState.getDimension();

            builder.startObject("cluster_aggregation")
                .field("total_shards", getSuccessfulShards())
                .field("dimension", dimensionCount)
                .startArray("dimensions");

            for (int dim = 0; dim < dimensionCount; dim++) {
                List<StatisticalSummary> dimensionStats = new ArrayList<>();

                // Collect statistics from all shards and segments for this dimension
                for (KNNIndexShardProfileResult shardResult : shardProfileResults) {
                    for (SegmentProfilerState state : shardResult.segmentProfilerStateList) {
                        if (dim < state.getStatistics().size()) {
                            dimensionStats.add(state.getStatistics().get(dim));
                        }
                    }
                }

                StatisticalSummaryValues aggregatedStats = AggregateSummaryStatistics.aggregate(dimensionStats);

                builder.startObject()
                    .field("dimension_id", dim)
                    .field("count", aggregatedStats.getN())
                    .field("min", aggregatedStats.getMin())
                    .field("max", aggregatedStats.getMax())
                    .field("mean", aggregatedStats.getMean())
                    .field("std_deviation", Math.sqrt(aggregatedStats.getVariance()))
                    .field("sum", aggregatedStats.getSum())
                    .field("variance", aggregatedStats.getVariance())
                    .endObject();
            }
            builder.endArray().endObject();
        }

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
