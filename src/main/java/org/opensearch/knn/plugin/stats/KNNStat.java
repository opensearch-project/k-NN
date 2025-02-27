/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import lombok.Getter;
import org.opensearch.core.action.ActionListener;

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
     * Allows stats to set context via asynchronous calls. This should only be used for cluster level stats.
     *
     * @param actionListener listener to call once the context is setup. If no async calls are made, then do nothing.
     * @return listener that will execute setup on response.
     */
    public ActionListener<Void> setupContext(ActionListener<Void> actionListener) {
        return actionListener;
    }

    /**
     * Get the value of the statistic
     *
     * @return value of the stat
     */
    public T getValue() {
        return supplier.get();
    }
}
