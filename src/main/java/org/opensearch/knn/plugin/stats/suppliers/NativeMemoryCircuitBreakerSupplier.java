/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import lombok.AllArgsConstructor;
import org.opensearch.knn.index.memory.breaker.NativeMemoryCircuitBreakerService;

import java.util.function.Supplier;

/**
 * Supplier for circuit breaker stats
 */
@AllArgsConstructor
public class NativeMemoryCircuitBreakerSupplier implements Supplier<Boolean> {

    private final NativeMemoryCircuitBreakerService nativeMemoryCircuitBreakerService;

    @Override
    public Boolean get() {
        return nativeMemoryCircuitBreakerService.isCircuitBreakerTriggered();
    }
}
