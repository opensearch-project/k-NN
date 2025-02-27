/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.HashSet;
import java.util.Set;

/**
 * Enum contains names of the stats
 */
public enum StatNames {
    HIT_COUNT("hit_count"),
    MISS_COUNT("miss_count"),
    LOAD_SUCCESS_COUNT("load_success_count"),
    LOAD_EXCEPTION_COUNT("load_exception_count"),
    TOTAL_LOAD_TIME("total_load_time"),
    EVICTION_COUNT("eviction_count"),
    GRAPH_MEMORY_USAGE("graph_memory_usage"),
    GRAPH_MEMORY_USAGE_PERCENTAGE("graph_memory_usage_percentage"),
    CACHE_CAPACITY_REACHED("cache_capacity_reached"),
    INDICES_IN_CACHE("indices_in_cache"),
    CIRCUIT_BREAKER_TRIGGERED("circuit_breaker_triggered"),
    MODEL_INDEX_STATUS("model_index_status"),
    FAISS_LOADED("faiss_initialized"),
    NMSLIB_LOADED("nmslib_initialized"),
    LUCENE_LOADED("lucene_initialized"),
    INDEXING_FROM_MODEL_DEGRADED("indexing_from_model_degraded"),
    GRAPH_QUERY_ERRORS(KNNCounter.GRAPH_QUERY_ERRORS.getName()),
    GRAPH_QUERY_REQUESTS(KNNCounter.GRAPH_QUERY_REQUESTS.getName()),
    GRAPH_INDEX_ERRORS(KNNCounter.GRAPH_INDEX_ERRORS.getName()),
    GRAPH_INDEX_REQUESTS(KNNCounter.GRAPH_INDEX_REQUESTS.getName()),
    KNN_QUERY_REQUESTS(KNNCounter.KNN_QUERY_REQUESTS.getName()),
    SCRIPT_COMPILATIONS(KNNCounter.SCRIPT_COMPILATIONS.getName()),
    SCRIPT_COMPILATION_ERRORS(KNNCounter.SCRIPT_COMPILATION_ERRORS.getName()),
    SCRIPT_QUERY_REQUESTS(KNNCounter.SCRIPT_QUERY_REQUESTS.getName()),
    TRAINING_REQUESTS(KNNCounter.TRAINING_REQUESTS.getName()),
    TRAINING_ERRORS(KNNCounter.TRAINING_ERRORS.getName()),
    TRAINING_MEMORY_USAGE("training_memory_usage"),
    TRAINING_MEMORY_USAGE_PERCENTAGE("training_memory_usage_percentage"),
    SCRIPT_QUERY_ERRORS(KNNCounter.SCRIPT_QUERY_ERRORS.getName()),
    KNN_QUERY_WITH_FILTER_REQUESTS(KNNCounter.KNN_QUERY_WITH_FILTER_REQUESTS.getName()),
    GRAPH_STATS("graph_stats"),
    REFRESH("refresh"),
    MERGE("merge"),
    REMOTE_VECTOR_INDEX_BUILD_STATS("remote_vector_index_build_stats"),
    CLIENT_STATS("client_stats"),
    REPOSITORY_STATS("repository_stats"),
    BUILD_STATS("build_stats"),
    MIN_SCORE_QUERY_REQUESTS(KNNCounter.MIN_SCORE_QUERY_REQUESTS.getName()),
    MIN_SCORE_QUERY_WITH_FILTER_REQUESTS(KNNCounter.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS.getName()),
    MAX_DISTANCE_QUERY_REQUESTS(KNNCounter.MAX_DISTANCE_QUERY_REQUESTS.getName()),
    MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS(KNNCounter.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS.getName());

    private String name;

    StatNames(String name) {
        this.name = name;
    }

    /**
     * Get stat name
     *
     * @return name
     */
    public String getName() {
        return name;
    }

    /**
     * Get all stat names
     *
     * @return set of all stat names
     */
    public static Set<String> getNames() {
        Set<String> names = new HashSet<>();

        for (StatNames statName : StatNames.values()) {
            names.add(statName.getName());
        }
        return names;
    }
}
