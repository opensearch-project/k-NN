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
public class KNNProfileResponse extends BroadcastResponse {
    private static final String FIELD_SHARD_PROFILES = "shard_profiles";
    private static final String FIELD_SEGMENTS = "segments";
    private static final String FIELD_SEGMENT_ID = "segment_id";
    private static final String FIELD_DIMENSION = "dimension";
    private static final String FIELD_VECTOR_STATISTICS = "vector_statistics";
    private static final String FIELD_DIMENSION_INDEX = "dimension_index";
    private static final String FIELD_STATISTICS = "statistics";
    private static final String FIELD_COUNT = "count";
    private static final String FIELD_MIN = "min";
    private static final String FIELD_MAX = "max";
    private static final String FIELD_SUM = "sum";
    private static final String FIELD_MEAN = "mean";
    private static final String FIELD_GEOMETRIC_MEAN = "geometric_mean";
    private static final String FIELD_VARIANCE = "variance";
    private static final String FIELD_STD_DEVIATION = "std_deviation";
    private static final String FIELD_SUM_OF_SQUARES = "sum_of_squares";
    private static final String FIELD_AGGREGATED = "aggregated";
    private static final String FIELD_TOTAL_SEGMENTS = "total_segments";
    private static final String FIELD_DIMENSIONS = "dimensions";
    private static final String FIELD_DIMENSION_ID = "dimension_id";
    private static final String FIELD_CLUSTER_AGGREGATION = "cluster_aggregation";
    private static final String FIELD_TOTAL_SHARDS = "total_shards";
    private static final String FIELD_FAILURES = "failures";

    List<KNNIndexShardProfileResult> shardProfileResults;

    public KNNProfileResponse() {}

    public KNNProfileResponse(StreamInput in) throws IOException {
        super(in);
        int size = in.readInt();
        List<KNNIndexShardProfileResult> results = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            results.add(new KNNIndexShardProfileResult(in));
        }
        this.shardProfileResults = results;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeInt(shardProfileResults.size());
        for (KNNIndexShardProfileResult result : shardProfileResults) {
            result.writeTo(out);
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
    protected void addCustomXContentFields(XContentBuilder builder, Params params) throws IOException {
        addShardProfiles(builder);
        addClusterAggregation(builder);
        addShardFailures(builder, params);
    }

    private void addShardProfiles(XContentBuilder builder) throws IOException {
        builder.startObject(FIELD_SHARD_PROFILES);
        for (KNNIndexShardProfileResult shardProfileResult : shardProfileResults) {
            builder.startObject(shardProfileResult.shardId);
            addSegmentStatistics(builder, shardProfileResult);
            addShardAggregatedStatistics(builder, shardProfileResult);
            builder.endObject();
        }
        builder.endObject();
    }

    private void addSegmentStatistics(XContentBuilder builder, KNNIndexShardProfileResult shardProfileResult) throws IOException {
        builder.startArray(FIELD_SEGMENTS);
        for (SegmentProfilerState state : shardProfileResult.segmentProfilerStateList) {
            builder.startObject()
                .field(FIELD_SEGMENT_ID, state.getSegmentId())
                .field(FIELD_DIMENSION, state.getDimension())
                .startArray(FIELD_VECTOR_STATISTICS);
            addDimensionStatistics(builder, state);
            builder.endArray().endObject();
        }
        builder.endArray();
    }

    private void addDimensionStatistics(XContentBuilder builder, SegmentProfilerState state) throws IOException {
        for (int i = 0; i < state.getStatistics().size(); i++) {
            SummaryStatistics stats = state.getStatistics().get(i);
            builder.startObject()
                .field(FIELD_DIMENSION_INDEX, i)
                .startObject(FIELD_STATISTICS)
                .field(FIELD_COUNT, stats.getN())
                .field(FIELD_MIN, stats.getMin())
                .field(FIELD_MAX, stats.getMax())
                .field(FIELD_SUM, stats.getSum())
                .field(FIELD_MEAN, stats.getMean())
                .field(FIELD_GEOMETRIC_MEAN, stats.getGeometricMean())
                .field(FIELD_VARIANCE, stats.getVariance())
                .field(FIELD_STD_DEVIATION, stats.getStandardDeviation())
                .field(FIELD_SUM_OF_SQUARES, stats.getSumsq())
                .endObject()
                .endObject();
        }
    }

    private void addShardAggregatedStatistics(XContentBuilder builder, KNNIndexShardProfileResult shardProfileResult) throws IOException {
        if (!shardProfileResult.segmentProfilerStateList.isEmpty()) {
            SegmentProfilerState firstState = shardProfileResult.segmentProfilerStateList.get(0);
            int dimensionCount = firstState.getDimension();

            builder.startObject(FIELD_AGGREGATED)
                .field(FIELD_TOTAL_SEGMENTS, shardProfileResult.segmentProfilerStateList.size())
                .field(FIELD_DIMENSION, dimensionCount)
                .startArray(FIELD_DIMENSIONS);

            addAggregatedDimensionStatistics(builder, shardProfileResult, dimensionCount);

            builder.endArray().endObject();
        }
    }

    private void addClusterAggregation(XContentBuilder builder) throws IOException {
        if (!shardProfileResults.isEmpty() && !shardProfileResults.get(0).segmentProfilerStateList.isEmpty()) {
            SegmentProfilerState firstState = shardProfileResults.get(0).segmentProfilerStateList.get(0);
            int dimensionCount = firstState.getDimension();

            builder.startObject(FIELD_CLUSTER_AGGREGATION)
                .field(FIELD_TOTAL_SHARDS, getSuccessfulShards())
                .field(FIELD_DIMENSION, dimensionCount)
                .startArray(FIELD_DIMENSIONS);

            addClusterDimensionStatistics(builder, dimensionCount);

            builder.endArray().endObject();
        }
    }

    private void addAggregatedDimensionStatistics(
        XContentBuilder builder,
        KNNIndexShardProfileResult shardProfileResult,
        int dimensionCount
    ) throws IOException {
        for (int dim = 0; dim < dimensionCount; dim++) {
            List<StatisticalSummary> dimensionStats = collectDimensionStats(shardProfileResult.segmentProfilerStateList, dim);
            addAggregatedStats(builder, dim, dimensionStats);
        }
    }

    private void addClusterDimensionStatistics(XContentBuilder builder, int dimensionCount) throws IOException {
        for (int dim = 0; dim < dimensionCount; dim++) {
            List<StatisticalSummary> dimensionStats = collectClusterDimensionStats(dim);
            addAggregatedStats(builder, dim, dimensionStats);
        }
    }

    private List<StatisticalSummary> collectDimensionStats(List<SegmentProfilerState> states, int dimension) {
        List<StatisticalSummary> stats = new ArrayList<>();
        for (SegmentProfilerState state : states) {
            if (dimension < state.getStatistics().size()) {
                stats.add(state.getStatistics().get(dimension));
            }
        }
        return stats;
    }

    private List<StatisticalSummary> collectClusterDimensionStats(int dimension) {
        List<StatisticalSummary> stats = new ArrayList<>();
        for (KNNIndexShardProfileResult shardResult : shardProfileResults) {
            for (SegmentProfilerState state : shardResult.segmentProfilerStateList) {
                if (dimension < state.getStatistics().size()) {
                    stats.add(state.getStatistics().get(dimension));
                }
            }
        }
        return stats;
    }

    private void addAggregatedStats(XContentBuilder builder, int dimension, List<StatisticalSummary> stats) throws IOException {
        StatisticalSummaryValues aggregatedStats = AggregateSummaryStatistics.aggregate(stats);
        builder.startObject()
            .field(FIELD_DIMENSION_ID, dimension)
            .field(FIELD_COUNT, aggregatedStats.getN())
            .field(FIELD_MIN, aggregatedStats.getMin())
            .field(FIELD_MAX, aggregatedStats.getMax())
            .field(FIELD_MEAN, aggregatedStats.getMean())
            .field(FIELD_STD_DEVIATION, Math.sqrt(aggregatedStats.getVariance()))
            .field(FIELD_SUM, aggregatedStats.getSum())
            .field(FIELD_VARIANCE, aggregatedStats.getVariance())
            .endObject();
    }

    private void addShardFailures(XContentBuilder builder, Params params) throws IOException {
        if (getShardFailures() != null && getShardFailures().length > 0) {
            builder.startArray(FIELD_FAILURES);
            for (DefaultShardOperationFailedException failure : getShardFailures()) {
                failure.toXContent(builder, params);
            }
            builder.endArray();
        }
    }
}
