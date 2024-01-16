/*
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.OpenSearchParseException;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsRequest;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.index.IndexModule;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryCacheManagerDto;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;
import org.opensearch.monitor.jvm.JvmInfo;
import org.opensearch.monitor.os.OsProbe;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.opensearch.core.common.unit.ByteSizeValue.parseBytesSizeValue;
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

    private static final Logger logger = LogManager.getLogger(KNNSettings.class);
    private static KNNSettings INSTANCE;
    private static final OsProbe osProbe = OsProbe.getInstance();

    private ClusterService clusterService;
    private Client client;

    private KNNSettings() {}

    public static synchronized KNNSettings state() {
        if (INSTANCE == null) {
            INSTANCE = new KNNSettings();
        }
        return INSTANCE;
    }

    public void initialize(Client client, ClusterService clusterService) {
        this.client = client;
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

    // TODO: Setters
    // Move to class that encapsulates the logic
    /**
     * Updates knn.circuit_breaker.triggered setting to true/false
     * @param flag true/false
     */
    public synchronized void updateCircuitBreakerSettings(boolean flag) {
        ClusterUpdateSettingsRequest clusterUpdateSettingsRequest = new ClusterUpdateSettingsRequest();
        Settings circuitBreakerSettings = Settings.builder().put(KNN_CIRCUIT_BREAKER_TRIGGERED, flag).build();
        clusterUpdateSettingsRequest.persistentSettings(circuitBreakerSettings);
        client.admin().cluster().updateSettings(clusterUpdateSettingsRequest, new ActionListener<ClusterUpdateSettingsResponse>() {
            @Override
            public void onResponse(ClusterUpdateSettingsResponse clusterUpdateSettingsResponse) {
                logger.debug(
                    "Cluster setting {}, acknowledged: {} ",
                    clusterUpdateSettingsRequest.persistentSettings(),
                    clusterUpdateSettingsResponse.isAcknowledged()
                );
            }

            @Override
            public void onFailure(Exception e) {
                logger.info(
                    "Exception while updating circuit breaker setting {} to {}",
                    clusterUpdateSettingsRequest.persistentSettings(),
                    e.getMessage()
                );
            }
        });
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

    // TODO Validators: Move to class where they are defined
    static class SpaceTypeValidator implements Setting.Validator<String> {

        @Override
        public void validate(String value) {
            try {
                SpaceType.getSpace(value);
            } catch (IllegalArgumentException ex) {
                throw new InvalidParameterException(ex.getMessage());
            }
        }
    }

    public void onIndexModule(IndexModule module) {
        module.addSettingsUpdateConsumer(INDEX_KNN_ALGO_PARAM_EF_SEARCH_SETTING, newVal -> {
            logger.debug("The value of [KNN] setting [{}] changed to [{}]", KNN_ALGO_PARAM_EF_SEARCH, newVal);
            // TODO: replace cache-rebuild with index reload into the cache
            NativeMemoryCacheManager.getInstance().rebuildCache();
        });
    }

    // TODO: Parsers/utility functions
    public static ByteSizeValue parseknnMemoryCircuitBreakerValue(String sValue, String settingName) {
        settingName = Objects.requireNonNull(settingName);
        if (sValue != null && sValue.endsWith("%")) {
            final String percentAsString = sValue.substring(0, sValue.length() - 1);
            try {
                final double percent = Double.parseDouble(percentAsString);
                if (percent < 0 || percent > 100) {
                    throw new OpenSearchParseException("percentage should be in [0-100], got [{}]", percentAsString);
                }
                long physicalMemoryInBytes = osProbe.getTotalPhysicalMemorySize();
                if (physicalMemoryInBytes <= 0) {
                    throw new IllegalStateException("Physical memory size could not be determined");
                }
                long esJvmSizeInBytes = JvmInfo.jvmInfo().getMem().getHeapMax().getBytes();
                long eligibleMemoryInBytes = physicalMemoryInBytes - esJvmSizeInBytes;
                return new ByteSizeValue((long) ((percent / 100) * eligibleMemoryInBytes), ByteSizeUnit.BYTES);
            } catch (NumberFormatException e) {
                throw new OpenSearchParseException("failed to parse [{}] as a double", e, percentAsString);
            }
        } else {
            return parseBytesSizeValue(sValue, settingName);
        }
    }

    public static String percentageAsString(Integer percentage) {
        return percentage + "%";
    }

    public static Double percentageAsFraction(Integer percentage) {
        return percentage / 100.0;
    }
}
