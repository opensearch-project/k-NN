/*
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryCacheManagerDto;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.opensearch.knn.index.KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD;
import static org.opensearch.knn.index.KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE;
import static org.opensearch.knn.index.KNNSettingsDefinitions.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.INDEX_KNN_ALGO_PARAM_EF_SEARCH_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.INDEX_KNN_ALGO_PARAM_M_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.INDEX_KNN_SPACE_TYPE;
import static org.opensearch.knn.index.KNNSettingsDefinitions.IS_KNN_INDEX_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_ALGO_PARAM_EF_SEARCH;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_ALGO_PARAM_INDEX_THREAD_QTY;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CACHE_ITEM_EXPIRY_ENABLED;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_TRIGGERED;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;
import static org.opensearch.knn.index.KNNSettingsDefinitions.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT;
import static org.opensearch.knn.index.KNNSettingsDefinitions.MODEL_CACHE_SIZE_LIMIT_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING;
import static org.opensearch.knn.index.KNNSettingsDefinitions.MODEL_INDEX_NUMBER_OF_SHARDS_SETTING;
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
        setSettingsUpdateConsumers();
    }

    public void setClusterService(ClusterService clusterService) {
        this.clusterService = clusterService;
    }

    // TODO: This should be added in constructor of the class where setting is defined. Examples:
    // 1. ClusterManagerTaskThrottler
    // 2. HierarchyCircuitBreakerService
    private void setSettingsUpdateConsumers() {
        clusterService.getClusterSettings().addSettingsUpdateConsumer(updatedSettings -> {
            // When any of the dynamic settings are updated, rebuild the cache with the updated values. Use the current
            // cluster settings values as defaults.
            NativeMemoryCacheManagerDto.NativeMemoryCacheManagerDtoBuilder builder = NativeMemoryCacheManagerDto.builder();

            builder.isWeightLimited(
                updatedSettings.getAsBoolean(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, getSettingValue(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED))
            );

            builder.maxWeight(((ByteSizeValue) getSettingValue(KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)).getKb());
            if (updatedSettings.hasValue(KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)) {
                builder.maxWeight(((ByteSizeValue) getSetting(KNN_MEMORY_CIRCUIT_BREAKER_LIMIT).get(updatedSettings)).getKb());
            }

            builder.isExpirationLimited(
                updatedSettings.getAsBoolean(KNN_CACHE_ITEM_EXPIRY_ENABLED, getSettingValue(KNN_CACHE_ITEM_EXPIRY_ENABLED))
            );

            builder.expiryTimeInMin(
                updatedSettings.getAsTime(KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES, getSettingValue(KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES))
                    .getMinutes()
            );

            NativeMemoryCacheManager.getInstance().rebuildCache(builder.build());
        }, new ArrayList<>(dynamicCacheSettings.values()));
    }

    // TODO: Getters
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

    public List<Setting<?>> getSettings() {
        List<Setting<?>> settings = Arrays.asList(
            INDEX_KNN_SPACE_TYPE,
            INDEX_KNN_ALGO_PARAM_M_SETTING,
            INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING,
            INDEX_KNN_ALGO_PARAM_EF_SEARCH_SETTING,
            KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING,
            KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING,
            KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING,
            IS_KNN_INDEX_SETTING,
            MODEL_INDEX_NUMBER_OF_SHARDS_SETTING,
            MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING,
            MODEL_CACHE_SIZE_LIMIT_SETTING,
            ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING
        );
        return Stream.concat(settings.stream(), dynamicCacheSettings.values().stream()).collect(Collectors.toList());
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
