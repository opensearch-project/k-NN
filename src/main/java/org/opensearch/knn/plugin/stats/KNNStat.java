/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import lombok.Getter;

import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

/**
 * Class represents a stat the plugin keeps track of
 */
public class KNNStat<T> {
    @Getter
    private final boolean isClusterLevel;
    private final Supplier<T> supplier;

    /**
     * Constructor
     *
     * @param isClusterLevel the scope of the stat
     * @param supplier supplier that returns the stat's value
     */
    public KNNStat(Boolean isClusterLevel, Supplier<T> supplier) {
        this.isClusterLevel = isClusterLevel;
        this.supplier = supplier;
    }

    /**
     * Allows a cluster stat to depend on node stats. This should only be set for cluster stats and should only return
     * node stats.
     *
     * @return list of dependent node stat names. Null if none
     */
    public List<String> dependentNodeStats() {
        return Collections.emptyList();
    }

    /**
     * Get the value of the statistic
     *
     * @return value of the stat
     */
    public T getValue() {
        return supplier.get();
    }

    /**
     * Get the value of the statistic potentially using the {@link KNNNodeStatAggregation}
     *
     * @param aggregation that can be used for cluster stats
     * @return value of the stat
     */
    public T getValue(KNNNodeStatAggregation aggregation) {
        return supplier.get();
    }
}
