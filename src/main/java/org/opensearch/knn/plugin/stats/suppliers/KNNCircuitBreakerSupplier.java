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