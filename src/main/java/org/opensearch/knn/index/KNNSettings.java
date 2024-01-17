/*
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;

import static org.opensearch.knn.index.KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD;
import static org.opensearch.knn.index.KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE;
import static org.opensearch.knn.index.KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_ALGO_PARAM_EF_SEARCH;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_ALGO_PARAM_INDEX_THREAD_QTY;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_TRIGGERED;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT;
import static org.opensearch.knn.index.KNNSettingsDefinitions.dynamicCacheSettings;

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

    public static boolean isKNNPluginEnabled() {
        return KNNSettings.state().getSettingValue(KNNSettingsDefinitions.KNN_PLUGIN_ENABLED);
    }

    public static boolean isCircuitBreakerTriggered() {
        return KNNSettings.state().getSettingValue(KNN_CIRCUIT_BREAKER_TRIGGERED);
    }

    public static ByteSizeValue getCircuitBreakerLimit() {
        return KNNSettings.state().getSettingValue(KNN_MEMORY_CIRCUIT_BREAKER_LIMIT);
    }

    public static double getCircuitBreakerUnsetPercentage() {
        return KNNSettings.state().getSettingValue(KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE);
    }

    public static Integer getFilteredExactSearchThreshold(final String indexName) {
        return KNNSettings.state().clusterService.state()
            .getMetadata()
            .index(indexName)
            .getSettings()
            .getAsInt(ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE);
    }

    /**
     *
     * @param index Name of the index
     * @return efSearch value
     */
    public static int getEfSearchParam(String index) {
        final IndexMetadata indexMetadata = KNNSettings.state().clusterService.state().getMetadata().index(index);
        return indexMetadata.getSettings()
            .getAsInt(KNN_ALGO_PARAM_EF_SEARCH, IndexHyperParametersUtil.getHNSWEFSearchValue(indexMetadata.getCreationVersion()));
    }
}
