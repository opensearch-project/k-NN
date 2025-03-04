/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class CircuitBreakerStatTests extends KNNTestCase {

    public void testGetValue() {
        CircuitBreakerStat stat = new CircuitBreakerStat();
        KNNNodeStatAggregation knnNodeStatAggregation = mock(KNNNodeStatAggregation.class);
        when(knnNodeStatAggregation.isClusterLevelCircuitBreakerTripped()).thenReturn(false);
        assertFalse(stat.getValue(knnNodeStatAggregation));
        when(knnNodeStatAggregation.isClusterLevelCircuitBreakerTripped()).thenReturn(true);
        assertTrue(stat.getValue(knnNodeStatAggregation));
    }
}
