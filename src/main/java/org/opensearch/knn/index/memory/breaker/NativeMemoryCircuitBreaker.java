/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory.breaker;

import com.google.common.annotations.VisibleForTesting;
import lombok.AllArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

/**
 * The circuit breaker gets tripped based on memory demand tracked by the {@link NativeMemoryCacheManager}.
 * When {@link NativeMemoryCacheManager}'s cache fills up, if the circuit breaking logic is enabled, it will trip the
 * circuit breaker. Elsewhere in the code, the circuit breaker's value can be queried to prevent actions that should
 * not happen during high memory pressure.
 */
@AllArgsConstructor
@Log4j2
public class NativeMemoryCircuitBreaker {
    private final KNNSettings knnSettings;

    /**
     * Checks if the circuit breaker is triggered
     *
     * @return true if circuit breaker is triggered; false otherwise
     */
    public boolean isTripped() {
        return knnSettings.getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED);
    }

    /**
     * Sets circuit breaker to new value
     *
     * @param circuitBreaker value to update circuit breaker to
     */
    public void set(boolean circuitBreaker) {
        knnSettings.updateBooleanSetting(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED, circuitBreaker);
    }

    /**
     * Gets the limit of the circuit breaker
     *
     * @return limit as ByteSizeValue of native memory circuit breaker
     */
    public ByteSizeValue getLimit() {
        return knnSettings.getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT);
    }

    /**
     * Determine if the circuit breaker is enabled
     *
     * @return true if circuit breaker is enabled. False otherwise.
     */
    public boolean isEnabled() {
        return knnSettings.getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED);
    }

    /**
     * Returns the percentage as a double for when to unset the circuit breaker
     *
     * @return percentage as double for unsetting circuit breaker
     */
    @VisibleForTesting
    double getUnsetPercentage() {
        return knnSettings.getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE);
    }
}
