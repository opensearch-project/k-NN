/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.Getter;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.threadpool.ThreadPool;

/**
 * Runs the circuit breaker logic and updates the settings
 */
@Getter
public class KNNCircuitBreaker {
    public static final String KNN_CIRCUIT_BREAKER_TIER = "knn_cb_tier";
    public static int CB_TIME_INTERVAL = 2 * 60; // seconds

    private static KNNCircuitBreaker INSTANCE;

    private boolean isTripped = false;

    private KNNCircuitBreaker() {}

    public static synchronized KNNCircuitBreaker getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new KNNCircuitBreaker();
        }
        return INSTANCE;
    }

    /**
     * Initialize the circuit breaker
     *
     * @param threadPool ThreadPool instance
     */
    public void initialize(ThreadPool threadPool) {
        NativeMemoryCacheManager nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        Runnable runnable = () -> {
            if (isTripped) {
                long currentSizeKiloBytes = nativeMemoryCacheManager.getCacheSizeInKilobytes();
                long circuitBreakerLimitSizeKiloBytes = KNNSettings.state().getCircuitBreakerLimit().getKb();
                long circuitBreakerUnsetSizeKiloBytes = (long) ((KNNSettings.getCircuitBreakerUnsetPercentage() / 100)
                    * circuitBreakerLimitSizeKiloBytes);

                // Unset capacityReached flag if currentSizeBytes is less than circuitBreakerUnsetSizeBytes
                if (currentSizeKiloBytes <= circuitBreakerUnsetSizeKiloBytes) {
                    setTripped(false);
                }
            }
        };
        threadPool.scheduleWithFixedDelay(runnable, TimeValue.timeValueSeconds(CB_TIME_INTERVAL), ThreadPool.Names.GENERIC);
    }

    /**
     * Set the circuit breaker flag
     *
     * @param isTripped true if circuit breaker is tripped, false otherwise
     */
    public synchronized void setTripped(boolean isTripped) {
        this.isTripped = isTripped;
    }
}
