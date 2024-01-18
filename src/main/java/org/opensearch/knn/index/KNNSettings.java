/*
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Setting;

import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_CIRCUIT_BREAKER_TRIGGERED;
import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING;
import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE;
import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.dynamicCacheSettings;
import static org.opensearch.knn.index.query.KNNWeight.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD;
import static org.opensearch.knn.index.query.KNNWeight.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING;
import static org.opensearch.knn.jni.JNIService.KNN_ALGO_PARAM_INDEX_THREAD_QTY;
import static org.opensearch.knn.jni.JNIService.KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING;

/**
 * This is a utility class for working with k-NN settings
 */
public class KNNSettings {

    private static KNNSettings INSTANCE;

    private ClusterService clusterService;

    private KNNSettings() {}

    public static synchronized KNNSettings state() {
        if (INSTANCE == null) {
            INSTANCE = new KNNSettings();
        }
        return INSTANCE;
    }

    public void initialize(ClusterService clusterService) {
        this.clusterService = clusterService;
    }

    // TODO: Utility get methods
    // In general, we could wrap these in some kind of util
    /**
     * Get setting value by key. Return default value if not configured explicitly.
     *
     * @param key   setting key.
     * @param <T> Setting type
     * @return T     setting value or default
     */
    @SuppressWarnings("unchecked")
    public <T> T getSettingValue(String key) {
        return (T) clusterService.getClusterSettings().get(getSetting(key));
    }

    /**
     * Get index metadata for a particular index
     * @param indexName
     * @return
     */
    public IndexMetadata getIndexMetadata(String indexName) {
        return KNNSettings.state().clusterService.state().getMetadata().index(indexName);
    }

    private Setting<?> getSetting(String key) {
        if (dynamicCacheSettings.containsKey(key)) {
            return dynamicCacheSettings.get(key);
        }

        if (KNN_CIRCUIT_BREAKER_TRIGGERED.equals(key)) {
            return KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING;
        }

        if (KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE.equals(key)) {
            return KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING;
        }

        if (KNN_ALGO_PARAM_INDEX_THREAD_QTY.equals(key)) {
            return KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING;
        }

        if (ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD.equals(key)) {
            return ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING;
        }

        throw new IllegalArgumentException("Cannot find setting by key [" + key + "]");
    }
}
