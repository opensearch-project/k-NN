/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import lombok.Getter;

import java.util.concurrent.atomic.LongAdder;

public enum KNNRemoteIndexBuildValue {

    // Repository Accumulating Stats
    WRITE_SUCCESS_COUNT("write_success_count"),
    WRITE_FAILURE_COUNT("write_failure_count"),
    WRITE_TIME("successful_write_time_in_millis"),
    READ_SUCCESS_COUNT("read_success_count"),
    READ_FAILURE_COUNT("read_failure_count"),
    READ_TIME("successful_read_time_in_millis"),

    // Remote Index Build Stats
    REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS("remote_index_build_current_merge_operations"),
    REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS("remote_index_build_current_flush_operations"),
    REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE("remote_index_build_current_merge_size"),
    REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE("remote_index_build_current_flush_size"),
    REMOTE_INDEX_BUILD_MERGE_TIME("remote_index_build_merge_time_in_millis"),
    REMOTE_INDEX_BUILD_FLUSH_TIME("remote_index_build_flush_time_in_millis"),

    // Client Stats
    BUILD_REQUEST_SUCCESS_COUNT("build_request_success_count"),
    BUILD_REQUEST_FAILURE_COUNT("build_request_failure_count"),
    STATUS_REQUEST_SUCCESS_COUNT("status_request_success_count"),
    STATUS_REQUEST_FAILURE_COUNT("status_request_failure_count"),
    INDEX_BUILD_SUCCESS_COUNT("index_build_success_count"),
    INDEX_BUILD_FAILURE_COUNT("index_build_failure_count"),
    WAITING_TIME("waiting_time_in_ms");

    @Getter
    private final String name;
    private final LongAdder value;

    /**
     * Constructor
     *
     * @param name name of the value
     */
    KNNRemoteIndexBuildValue(String name) {
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
     *
     * @param delta The amount to increment
     */
    public void incrementBy(long delta) {
        value.add(delta);
    }

    /**
     * Decrement the value by a specified amount
     *
     * @param delta The amount to decrement
     */
    public void decrementBy(long delta) {
        value.add(delta * -1);
    }

    /**
     * Reset the value to 0L.
     */
    public void reset() {
        value.reset();
    }
}
