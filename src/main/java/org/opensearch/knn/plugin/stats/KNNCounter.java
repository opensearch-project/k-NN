/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Contains a map of counters to keep track of different values
 */
public enum KNNCounter {
    GRAPH_QUERY_ERRORS("graph_query_errors"),
    GRAPH_QUERY_REQUESTS("graph_query_requests"),
    GRAPH_INDEX_ERRORS("graph_index_errors"),
    GRAPH_INDEX_REQUESTS("graph_index_requests"),
    KNN_QUERY_REQUESTS("knn_query_requests"),
    SCRIPT_COMPILATIONS("script_compilations"),
    SCRIPT_COMPILATION_ERRORS("script_compilation_errors"),
    SCRIPT_QUERY_REQUESTS("script_query_requests"),
    SCRIPT_QUERY_ERRORS("script_query_errors"),
    TRAINING_REQUESTS("training_requests"),
    TRAINING_ERRORS("training_errors"),
    KNN_QUERY_WITH_FILTER_REQUESTS("knn_query_with_filter_requests"),
    MIN_SCORE_QUERY_REQUESTS("min_score_query_requests"),
    MIN_SCORE_QUERY_WITH_FILTER_REQUESTS("min_score_query_with_filter_requests"),
    MAX_DISTANCE_QUERY_REQUESTS("max_distance_query_requests"),
    MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS("max_distance_query_with_filter_requests");

    private String name;
    private AtomicLong count;

    /**
     * Constructor
     *
     * @param name name of the counter
     */
    KNNCounter(String name) {
        this.name = name;
        this.count = new AtomicLong(0);
    }

    /**
     * Get name of counter
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
    public Long getCount() {
        return count.get();
    }

    /**
     * Increment the value of a counter
     */
    public void increment() {
        count.getAndIncrement();
    }

    /**
     * @param value counter value
     * Set the value of a counter
     */
    public void set(long value) {
        count.set(value);
    }
}
