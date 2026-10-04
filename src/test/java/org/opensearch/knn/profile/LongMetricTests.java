/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import org.opensearch.knn.KNNTestCase;

import java.util.Map;

public class LongMetricTests extends KNNTestCase {

    public void testConstructorInitializesValueToZero() {
        LongMetric metric = new LongMetric("test_metric");

        assertEquals("test_metric", metric.getName());
        assertEquals(0L, metric.getValue().longValue());
    }

    public void testSetAndGetValue() {
        LongMetric metric = new LongMetric("cardinality");
        metric.setValue(42L);

        assertEquals(42L, metric.getValue().longValue());
    }

    public void testToBreakdownMap() {
        LongMetric metric = new LongMetric("cardinality");
        metric.setValue(100L);

        Map<String, Long> breakdown = metric.toBreakdownMap();

        assertEquals(1, breakdown.size());
        assertEquals(100L, breakdown.get("cardinality").longValue());
    }

    public void testToBreakdownMap_withZeroValue() {
        LongMetric metric = new LongMetric("num_nested_docs");

        Map<String, Long> breakdown = metric.toBreakdownMap();

        assertEquals(0L, breakdown.get("num_nested_docs").longValue());
    }
}
