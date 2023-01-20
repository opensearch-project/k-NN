/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.HashMap;
import java.util.Map;

/**
 * Class represents all stats the plugin keeps track of
 */
public class KNNStats {

    private final Map<String, KNNStat<?>> knnStats;

    /**
     * Constructor
     *
     * @param knnStats Map that maps name of stat to KNNStat object
     */
    public KNNStats(Map<String, KNNStat<?>> knnStats) {
        this.knnStats = knnStats;
    }

    /**
     * Get the stats
     *
     * @return all of the stats
     */
    public Map<String, KNNStat<?>> getStats() {
        return knnStats;
    }

    /**
     * Get a map of the stats that are kept at the node level
     *
     * @return Map of stats kept at the node level
     */
    public Map<String, KNNStat<?>> getNodeStats() {
        return getClusterOrNodeStats(false);
    }

    /**
     * Get a map of the stats that are kept at the cluster level
     *
     * @return Map of stats kept at the cluster level
     */
    public Map<String, KNNStat<?>> getClusterStats() {
        return getClusterOrNodeStats(true);
    }

    private Map<String, KNNStat<?>> getClusterOrNodeStats(Boolean getClusterStats) {
        Map<String, KNNStat<?>> statsMap = new HashMap<>();

        for (Map.Entry<String, KNNStat<?>> entry : knnStats.entrySet()) {
            if (entry.getValue().isClusterLevel() == getClusterStats) {
                statsMap.put(entry.getKey(), entry.getValue());
            }
        }
        return statsMap;
    }
}
