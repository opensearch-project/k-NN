/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

public class DimensionStatisticAggregatorTests extends OpenSearchTestCase {

    private DimensionStatisticAggregator aggregator;
    private static final String TEST_SEGMENT_ID = "test-segment";

    @Before
    public void setUp() throws Exception {
        super.setUp();
        aggregator = new DimensionStatisticAggregator(0);
    }

    @Test
    public void testAddValues() {
        aggregator.addValue(TEST_SEGMENT_ID, 1.0f);
        aggregator.addValue(TEST_SEGMENT_ID, 2.0f);
        aggregator.addValue(TEST_SEGMENT_ID, 3.0f);

        assertEquals(1, aggregator.getSegmentStatistics().size());

        SummaryStatistics stats = aggregator.getSegmentStatistic(TEST_SEGMENT_ID);
        assertEquals(3, stats.getN());
        assertEquals(2.0, stats.getMean(), 0.001);
    }

    @Test
    public void testMultipleSegments() {
        String segment1 = "segment1";
        String segment2 = "segment2";

        aggregator.addValue(segment1, 1.0f);
        aggregator.addValue(segment1, 2.0f);
        aggregator.addValue(segment2, 4.0f);
        aggregator.addValue(segment2, 5.0f);

        Map<String, SummaryStatistics> segmentStats = aggregator.getSegmentStatistics();
        assertEquals(2, segmentStats.size());

        assertEquals(1.5, aggregator.getSegmentStatistic(segment1).getMean(), 0.001);
        assertEquals(4.5, aggregator.getSegmentStatistic(segment2).getMean(), 0.001);
    }

    @Test
    public void testAggregateStatistics() {
        aggregator.addValue("segment1", 1.0f);
        aggregator.addValue("segment1", 2.0f);
        aggregator.addValue("segment2", 3.0f);

        assertEquals(2.0, aggregator.getAggregateStats().getMean(), 0.001);
        assertEquals(1.0, aggregator.getAggregateStats().getStandardDeviation(), 0.001);
    }

    @Test
    public void testGetDimension() {
        assertEquals(0, aggregator.getDimensionId());
    }

    @Test
    public void testNonExistentSegment() {
        assertNull(aggregator.getSegmentStatistic("non-existent"));
    }
}
