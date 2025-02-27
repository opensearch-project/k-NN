/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.Getter;

/**
 * Runs the circuit breaker logic and updates the settings
 */
@Getter
public class KNNCircuitBreaker {
    public static final String KNN_CIRCUIT_BREAKER_TIER = "knn_cb_tier";

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
     * Set the circuit breaker flag
     *
     * @param isTripped true if circuit breaker is tripped, false otherwise
     */
    public synchronized void setTripped(boolean isTripped) {
        this.isTripped = isTripped;
    }
}
