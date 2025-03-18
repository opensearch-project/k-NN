/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.remoteindexbuild.stats;

import lombok.Getter;

import java.util.concurrent.atomic.LongAdder;

public enum RemoteIndexClientValue {
    BUILD_REQUEST_SUCCESS_COUNT("build_request_success_count"),
    BUILD_REQUEST_FAILURE_COUNT("build_request_failure_count"),
    STATUS_REQUEST_SUCCESS_COUNT("status_request_success_count"),
    STATUS_REQUEST_FAILURE_COUNT("status_request_failure_count");

    @Getter
    private final String name;
    private final LongAdder value;

    /**
     * Constructor
     * @param name name of the value
     */
    RemoteIndexClientValue(String name) {
        this.name = name;
        this.value = new LongAdder();
    }

    /**
     * Get the value
     * @return value
     */
    public Long getValue() {
        return value.longValue();
    }

    /**
     * Increment the value
     */
    public void increment() {
        value.increment();
    }

    /**
     * Decrement the value
     */
    public void decrement() {
        value.decrement();
    }

    /**
     * Increment the value by a specified amount
     * @param delta The amount to increment
     */
    public void incrementBy(long delta) {
        value.add(delta);
    }

    /**
     * Decrement the value by a specified amount
     * @param delta The amount to decrement
     */
    public void decrementBy(long delta) {
        value.add(delta * -1);
    }
}
