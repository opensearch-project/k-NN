/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Contains a map of counters to keep track of different values
 */
public enum KNNGraphValue {

    REFRESH_TOTAL_OPERATIONS("total"),
    REFRESH_TOTAL_TIME_IN_MILLIS("total_time_in_millis"),
    MERGE_CURRENT_OPERATIONS("current"),
    MERGE_CURRENT_DOCS("current_docs"),
    MERGE_CURRENT_SIZE_IN_BYTES("current_size_in_bytes"),
    MERGE_TOTAL_OPERATIONS("total"),
    MERGE_TOTAL_TIME_IN_MILLIS("total_time_in_millis"),
    MERGE_TOTAL_DOCS("total_docs"),
    MERGE_TOTAL_SIZE_IN_BYTES("total_size_in_bytes");

    private String name;
    private AtomicLong value;

    /**
     * Constructor
     *
     * @param name name of the counter
     */
    KNNGraphValue(String name) {
        this.name = name;
        this.value = new AtomicLong(0);
    }

    /**
     * Get name of value
     *
     * @return name
     */
    public String getName() {
        return name;
    }

    /**
     * Get the value of count
     *
     * @return count
     */
    public Long getValue() {
        return value.get();
    }

    /**
     * Increment the value of a counter
     */
    public void increment() {
        value.getAndIncrement();
    }

    /**
     * @param value counter value
     * Set the value of a counter
     */
    public void set(long value) {
        this.value.set(value);
    }
}
