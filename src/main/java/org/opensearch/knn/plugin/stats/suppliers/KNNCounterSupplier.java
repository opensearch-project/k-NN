/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
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