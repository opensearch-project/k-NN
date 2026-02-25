/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import org.opensearch.knn.index.KNNSettings;

import java.util.function.Supplier;

/**
 * Supplier for circuit breaker stats
 */
public class KNNCircuitBreakerSupplier implements Supplier<Boolean> {

    /**
     * Constructor
     */
    public KNNCircuitBreakerSupplier() {}

    @Override
    public Boolean get() {
        return KNNSettings.isCircuitBreakerTriggered();
    }
}
