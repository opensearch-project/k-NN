/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import lombok.Getter;
import org.opensearch.core.action.ActionListener;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Class represents a stat the plugin keeps track of
 */
public class KNNStat<T> {
    @Getter
    private Boolean isClusterLevel;
    private final Function<KNNStatFetchContext, T> statFetcher;

    /**
     * Constructor
     *
     * @param isClusterLevel the scope of the stat
     * @param supplier supplier that returns the stat's value
     */
    public KNNStat(Boolean isClusterLevel, Supplier<T> supplier) {
        this(isClusterLevel, context -> supplier.get());
    }

    public KNNStat(Boolean isClusterLevel, Function<KNNStatFetchContext, T> statFetcher) {
        this.isClusterLevel = isClusterLevel;
        this.statFetcher = statFetcher;
    }

    /**
     * Determines whether the stat is kept at the cluster level or the node level
     *
     * @return boolean that is true if the stat is clusterLevel; false otherwise
     */
    public Boolean isClusterLevel() {
        return isClusterLevel;
    }

    public ActionListener<Void> setupContext(KNNStatFetchContext knnStatFetchContext, ActionListener<Void> actionListener) {
        return actionListener;
    }

    /**
     * Get the value of the statistic
     *
     * @return value of the stat
     */
    public T getValue() {
        return getValue(null);
    }

    /**
     * Get the value of the statistic
     *
     * @param statFetchContext context for fetching the stat
     * @return value of the stat
     */
    public T getValue(KNNStatFetchContext statFetchContext) {
        return statFetcher.apply(statFetchContext);
    }
}
