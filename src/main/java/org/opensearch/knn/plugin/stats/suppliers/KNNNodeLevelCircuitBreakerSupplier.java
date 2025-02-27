/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import org.opensearch.knn.index.KNNCircuitBreaker;

import java.util.function.Supplier;

/**
 * Supplier for circuit breaker stats
 */
public class KNNNodeLevelCircuitBreakerSupplier implements Supplier<Boolean> {

    /**
     * Constructor
     */
    public KNNNodeLevelCircuitBreakerSupplier() {}

    @Override
    public Boolean get() {
        return KNNCircuitBreaker.getInstance().isTripped();
    }
}
