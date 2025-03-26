/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

public class DimensionStatisticAggregatorTests extends OpenSearchTestCase {

    private DimensionStatisticAggregator aggregator;
    private Collection<Float> testValues;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        aggregator = new DimensionStatisticAggregator(0);
        testValues = Arrays.asList(1.0f, 2.0f, 3.0f);
    }

    @Test
    public void testAddSegmentStatistics() {
        aggregator.addSegmentStatistics(testValues);
        assertEquals(1, aggregator.getSegmentStatistics().size());
    }

    @Test
    public void testMultipleSegments() {
        aggregator.addSegmentStatistics(testValues);
        aggregator.addSegmentStatistics(Arrays.asList(4.0f, 5.0f, 6.0f));

        Map<String, SummaryStatistics> segmentStats = aggregator.getSegmentStatistics();
        assertEquals(2, segmentStats.size());
    }

    @Test
    public void testAggregateStatistics() {
        aggregator.addSegmentStatistics(testValues);

        assertEquals(2.0, aggregator.getAggregateStatistics().getMean(), 0.001);
        assertEquals(1.0, aggregator.getAggregateStatistics().getStandardDeviation(), 0.001);
    }

    @Test
    public void testEmptyValues() {
        aggregator.addSegmentStatistics(Arrays.asList());
        assertEquals(0, aggregator.getAggregateStatistics().getN());
    }

    @Test
    public void testMultipleSegmentsAggregation() {
        aggregator.addSegmentStatistics(Arrays.asList(1.0f, 2.0f));
        aggregator.addSegmentStatistics(Arrays.asList(3.0f, 4.0f));

        assertEquals(2.5, aggregator.getAggregateStatistics().getMean(), 0.001);
    }
}
