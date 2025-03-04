/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import lombok.Getter;
import org.opensearch.knn.plugin.transport.KNNStatsNodeResponse;

import java.util.List;

/**
 * Class contains aggregations of node stats that can be used as cluster stats. For instance, for the circuit breaker
 * stat, we want to check if any nodes in the cluster have their circuit breaker tripped. This is reported as the
 * cache_capacity_reached stat. We need to perform an aggregation on this stat to check if the circuit breaker is
 * tripped for the cluster. We cannot rely on the node stats returned for this because some nodes may be filtered out
 */
public class KNNNodeStatAggregation {
    @Getter
    private final boolean isClusterLevelCircuitBreakerTripped;

    /**
     * From a set of node responses, create the aggregate stats
     *
     * @param nodeResponses
     */
    public KNNNodeStatAggregation(List<KNNStatsNodeResponse> nodeResponses) {
        this.isClusterLevelCircuitBreakerTripped = nodeResponses.stream().anyMatch(r -> {
            if (r == null) {
                return false;
            }

            if (r.getStatsMap() == null) {
                return false;
            }

            if (r.getStatsMap().containsKey(StatNames.CACHE_CAPACITY_REACHED.getName()) == false) {
                return false;
            }

            return (boolean) r.getStatsMap().get(StatNames.CACHE_CAPACITY_REACHED.getName());
        });
    }
}
