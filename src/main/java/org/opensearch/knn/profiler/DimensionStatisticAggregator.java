/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.profiler;

import lombok.Getter;
import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.Map;
import java.util.HashMap;

/**
 * Aggregates statistics for a specific dimension across multiple segments.
 * This class is used to collect and analyze statistical data for a particular dimension
 * across multiple segments of an index. This class helps to aggregate statistics across
 * different portions of the index.
 */
@Getter
public class DimensionStatisticAggregator {
    // The index/position of this dimension in the vector
    private final int dimensionId;
    private final AggregateSummaryStatistics aggregateStats;
    private final Map<String, SummaryStatistics> segmentToSummaryMapping;

    public DimensionStatisticAggregator(final int dimensionId) {
        this.dimensionId = dimensionId;
        this.aggregateStats = new AggregateSummaryStatistics();
        this.segmentToSummaryMapping = new HashMap<>();
    }

    /**
     * Adds a single value to the specified segment's statistics
     *
     * @param segmentId the identifier for the segment
     * @param value the value to add to the statistics
     */
    public void addValue(final String segmentId, final float value) {
        SummaryStatistics segmentStats = segmentToSummaryMapping.computeIfAbsent(
            segmentId,
            k -> aggregateStats.createContributingStatistics()
        );
        segmentStats.addValue(value);
    }

    /**
     * @return a map of segment IDs to their respective statistics
     */
    public Map<String, SummaryStatistics> getSegmentStatistics() {
        return new HashMap<>(segmentToSummaryMapping);
    }

    /**
     * Retrieves statistics for a specific segment
     *
     * @param segmentId the identifier for the segment
     * @return the statistics for the specified segment, or null if not found
     */
    public SummaryStatistics getSegmentStatistic(final String segmentId) {
        return segmentToSummaryMapping.get(segmentId);
    }

    /**
     * @return the dimension index this aggregator is responsible for
     */
    public int getDimensionId() {
        return dimensionId;
    }
}
