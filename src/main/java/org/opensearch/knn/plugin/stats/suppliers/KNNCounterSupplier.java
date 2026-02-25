/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import org.opensearch.knn.plugin.stats.KNNCounter;

import java.util.function.Supplier;

/**
 * Supplier for stats that need to keep count
 */
public class KNNCounterSupplier implements Supplier<Long> {
    private KNNCounter knnCounter;

    /**
     * Constructor
     *
     * @param knnCounter KNN Plugin Counter
     */
    public KNNCounterSupplier(KNNCounter knnCounter) {
        this.knnCounter = knnCounter;
    }

    @Override
    public Long get() {
        return knnCounter.getCount();
    }
}
