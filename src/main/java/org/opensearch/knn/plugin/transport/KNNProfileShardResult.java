/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.index.shard.ShardId;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.plugin.transport.KNNWarmupTransportAction.logger;

/**
 * Shard-level result for KNN profiling
 */
public class KNNProfileShardResult implements Writeable {
    private final ShardId shardId;
    private final List<StatisticalSummaryValues> dimensionStats;

    /**
     * Constructor
     * @param shardId the shard ID
     * @param dimensionStats statistical summaries for each dimension
     */
    public KNNProfileShardResult(ShardId shardId, List<StatisticalSummaryValues> dimensionStats) {
        this.shardId = shardId;
        this.dimensionStats = dimensionStats;
        logger.info("[KNN] Created KNNProfileShardResult for shard {} with {} stats", shardId, this.dimensionStats.size());
    }

    /**
     * Constructor for serialization
     * @param in stream input
     * @throws IOException if there's an error reading from the stream
     */
    public KNNProfileShardResult(StreamInput in) throws IOException {
        this.shardId = new ShardId(in);
        int size = in.readVInt();
        this.dimensionStats = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            double mean = in.readDouble();
            double variance = in.readDouble();
            long n = in.readVLong();
            double min = in.readDouble();
            double max = in.readDouble();
            double sum = in.readDouble();
            this.dimensionStats.add(new StatisticalSummaryValues(mean, variance, n, min, max, sum));
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        shardId.writeTo(out);
        out.writeVInt(dimensionStats.size());
        for (StatisticalSummaryValues stats : dimensionStats) {
            out.writeDouble(stats.getMean());
            out.writeDouble(stats.getVariance());
            out.writeVLong(stats.getN());
            out.writeDouble(stats.getMin());
            out.writeDouble(stats.getMax());
            out.writeDouble(stats.getSum());
        }
    }

    /**
     * Get the shard ID
     * @return shard ID
     */
    public ShardId getShardId() {
        return shardId;
    }

    /**
     * Get the statistical summaries for each dimension
     * @return list of statistical summaries
     */
    public List<StatisticalSummaryValues> getDimensionStats() {
        return dimensionStats;
    }
}
