/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.List;

/**
 * Cluster stat that checks if the circuit breaker is enabled. For clusters on or after version 3.0, this stat will
 * be populated by broadcasting a transport call to all nodes to see if any of their circuit breakers are set. Before
 * 3.0, it checks the cluster setting.
 */
public class CircuitBreakerStat extends KNNStat<Boolean> {

    public CircuitBreakerStat() {
        super(true, null);
    }

    @Override
    public List<String> dependentNodeStats() {
        return List.of(StatNames.CACHE_CAPACITY_REACHED.getName());
    }

    @Override
    public Boolean getValue() {
        return false;
    }

    @Override
    public Boolean getValue(KNNNodeStatAggregation aggregation) {
        if (aggregation == null) {
            return false;
        }
        return aggregation.isClusterLevelCircuitBreakerTripped();
    }
}
