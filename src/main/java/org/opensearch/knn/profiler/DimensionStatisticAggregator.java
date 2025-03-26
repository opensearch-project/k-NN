/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.descriptive.StatisticalSummary;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class DimensionStatisticAggregator {
    private final AggregateSummaryStatistics aggregateStats;
    private final Map<String, SummaryStatistics> segmentToSummaryMapping;
    private final int dimension;

    public DimensionStatisticAggregator(int dimension) {
        this.dimension = dimension;
        this.aggregateStats = new AggregateSummaryStatistics();
        this.segmentToSummaryMapping = new HashMap<>();
    }

    public void addSegmentStatistics(Collection<Float> values) {
        String segmentId = UUID.randomUUID().toString();
        SummaryStatistics segmentStats = aggregateStats.createContributingStatistics();

        for (Float value : values) {
            segmentStats.addValue(value);
        }

        segmentToSummaryMapping.put(segmentId, segmentStats);
    }

    public StatisticalSummary getAggregateStatistics() {
        return aggregateStats;
    }

    public Map<String, SummaryStatistics> getSegmentStatistics() {
        return new HashMap<>(segmentToSummaryMapping);
    }
}
