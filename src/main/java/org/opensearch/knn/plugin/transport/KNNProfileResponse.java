/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues;
import org.opensearch.action.support.broadcast.BroadcastResponse;
import org.opensearch.core.action.support.DefaultShardOperationFailedException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * Response for KNN profile request
 */
@Log4j2
public class KNNProfileResponse extends BroadcastResponse implements ToXContentObject {
    private final List<KNNProfileShardResult> shardResults;

    /**
     * Constructor
     */
    public KNNProfileResponse(
        int totalShards,
        int successfulShards,
        int failedShards,
        List<KNNProfileShardResult> shardResults,
        List<DefaultShardOperationFailedException> shardFailures
    ) {
        super(totalShards, successfulShards, failedShards, shardFailures);
        this.shardResults = shardResults != null ? shardResults : List.of();
    }

    /**
     * Constructor for serialization
     */
    public KNNProfileResponse(StreamInput in) throws IOException {
        super(in);
        int size = in.readVInt();
        shardResults = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            shardResults.add(new KNNProfileShardResult(in));
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeVInt(shardResults.size());
        for (KNNProfileShardResult result : shardResults) {
            result.writeTo(out);
        }
    }

    /**
     * Get aggregated dimension statistics by index
     */
    private Map<String, Map<Integer, Map<String, Object>>> getAggregatedStats() {
        Map<String, Map<Integer, Map<String, Object>>> indexDimensions = new HashMap<>();

        for (KNNProfileShardResult result : shardResults) {
            String indexName = result.getShardId().getIndexName();
            List<StatisticalSummaryValues> stats = result.getDimensionStats();

            if (stats == null || stats.isEmpty()) {
                continue;
            }

            Map<Integer, Map<String, Object>> dimensions = indexDimensions.computeIfAbsent(indexName, k -> new HashMap<>());

            for (int i = 0; i < stats.size(); i++) {
                StatisticalSummaryValues stat = stats.get(i);
                if (stat == null) {
                    continue;
                }

                int finalI = i;
                Map<String, Object> dimension = dimensions.computeIfAbsent(i, k -> {
                    Map<String, Object> newDim = new HashMap<>();
                    newDim.put("dimension", finalI);
                    newDim.put("count", 0L);
                    newDim.put("sum", 0.0);
                    return newDim;
                });

                long oldCount = (Long) dimension.get("count");
                double oldSum = (Double) dimension.get("sum");
                long newCount = oldCount + stat.getN();
                double newSum = oldSum + stat.getSum();

                dimension.put("count", newCount);
                dimension.put("sum", newSum);
                dimension.put("mean", newCount > 0 ? newSum / newCount : 0);

                if (!dimension.containsKey("min") || stat.getMin() < (Double) dimension.get("min")) {
                    dimension.put("min", stat.getMin());
                }
                if (!dimension.containsKey("max") || stat.getMax() > (Double) dimension.get("max")) {
                    dimension.put("max", stat.getMax());
                }

                if (newCount > 0) {
                    dimension.put("variance", stat.getVariance());
                    dimension.put("std_deviation", Math.sqrt(stat.getVariance()));
                }
            }
        }

        return indexDimensions;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field("total_shards", getTotalShards());
        builder.field("successful_shards", getSuccessfulShards());
        builder.field("failed_shards", getFailedShards());

        builder.startObject("profile_results");

        try {
            Map<String, Map<Integer, Map<String, Object>>> indexDimensions = getAggregatedStats();

            for (Map.Entry<String, Map<Integer, Map<String, Object>>> indexEntry : indexDimensions.entrySet()) {
                String indexName = indexEntry.getKey();
                Map<Integer, Map<String, Object>> dimensions = indexEntry.getValue();

                builder.startObject(indexName);
                builder.startArray("dimensions");

                List<Integer> dimensionIndices = new ArrayList<>(dimensions.keySet());
                Collections.sort(dimensionIndices);

                for (Integer dimIndex : dimensionIndices) {
                    builder.map(dimensions.get(dimIndex));
                }

                builder.endArray();
                builder.endObject();
            }

        } catch (Exception e) {
            log.error("Error generating profile results", e);
        }

        builder.endObject();

        if (getShardFailures() != null && getShardFailures().length > 0) {
            builder.startArray("failures");
            for (DefaultShardOperationFailedException failure : getShardFailures()) {
                builder.startObject();
                failure.toXContent(builder, params);
                builder.endObject();
            }
            builder.endArray();
        }

        builder.endObject();
        return builder;
    }
}
