/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.extern.log4j.Log4j2;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.OpenSearchParseException;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsRequest;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.index.IndexModule;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryCacheManagerDto;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;
import org.opensearch.monitor.jvm.JvmInfo;
import org.opensearch.monitor.os.OsProbe;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.opensearch.common.settings.Setting.Property.Dynamic;
import static org.opensearch.common.settings.Setting.Property.IndexScope;
import static org.opensearch.common.settings.Setting.Property.NodeScope;
import static org.opensearch.core.common.unit.ByteSizeValue.parseBytesSizeValue;
import static org.opensearch.common.unit.MemorySizeValue.parseBytesSizeValueOrHeapRatio;

/**
 * This class defines
 * 1. KNN settings to hold the <a href="https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md">HNSW algorithm parameters</a>.
 * 2. KNN settings to enable/disable plugin, circuit breaker settings
 * 3. KNN settings to manage graphs loaded in native memory
 */
@Log4j2
public class KNNSettings {

    private static final Logger logger = LogManager.getLogger(KNNSettings.class);
    private static KNNSettings INSTANCE;
    private static final OsProbe osProbe = OsProbe.getInstance();

    private static final int INDEX_THREAD_QTY_MAX = 32;

    /**
     * Settings name
     */
    public static final String KNN_SPACE_TYPE = "index.knn.space_type";
    public static final String KNN_ALGO_PARAM_M = "index.knn.algo_param.m";
    public static final String KNN_ALGO_PARAM_EF_CONSTRUCTION = "index.knn.algo_param.ef_construction";
    public static final String KNN_ALGO_PARAM_EF_SEARCH = "index.knn.algo_param.ef_search";
    public static final String KNN_ALGO_PARAM_INDEX_THREAD_QTY = "knn.algo_param.index_thread_qty";
    public static final String KNN_MEMORY_CIRCUIT_BREAKER_ENABLED = "knn.memory.circuit_breaker.enabled";
    public static final String KNN_MEMORY_CIRCUIT_BREAKER_LIMIT = "knn.memory.circuit_breaker.limit";
    public static final String KNN_VECTOR_STREAMING_MEMORY_LIMIT_IN_MB = "knn.vector_streaming_memory.limit";
    public static final String KNN_CIRCUIT_BREAKER_TRIGGERED = "knn.circuit_breaker.triggered";
    public static final String KNN_CACHE_ITEM_EXPIRY_ENABLED = "knn.cache.item.expiry.enabled";
    public static final String KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES = "knn.cache.item.expiry.minutes";
    public static final String KNN_PLUGIN_ENABLED = "knn.plugin.enabled";
    public static final String KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE = "knn.circuit_breaker.unset.percentage";
    public static final String KNN_INDEX = "index.knn";
    public static final String MODEL_INDEX_NUMBER_OF_SHARDS = "knn.model.index.number_of_shards";
    public static final String MODEL_INDEX_NUMBER_OF_REPLICAS = "knn.model.index.number_of_replicas";
    public static final String MODEL_CACHE_SIZE_LIMIT = "knn.model.cache.size.limit";
    public static final String ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD = "index.knn.advanced.filtered_exact_search_threshold";
    public static final String KNN_FAISS_AVX2_DISABLED = "knn.faiss.avx2.disabled";

    /**
     * Default setting values
     */
    public static final boolean KNN_DEFAULT_FAISS_AVX2_DISABLED_VALUE = false;
    public static final String INDEX_KNN_DEFAULT_SPACE_TYPE = "l2";
    public static final String INDEX_KNN_DEFAULT_SPACE_TYPE_FOR_BINARY = "hamming";
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_M = 16;
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH = 100;
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION = 100;
    public static final Integer KNN_DEFAULT_ALGO_PARAM_INDEX_THREAD_QTY = 1;
    public static final Integer KNN_DEFAULT_CIRCUIT_BREAKER_UNSET_PERCENTAGE = 75;
    public static final Integer KNN_DEFAULT_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE = 10; // By default, set aside 10% of the JVM for the limit
    public static final Integer KNN_MAX_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE = 25; // Model cache limit cannot exceed 25% of the JVM heap
    public static final String KNN_DEFAULT_MEMORY_CIRCUIT_BREAKER_LIMIT = "50%";
    public static final String KNN_DEFAULT_VECTOR_STREAMING_MEMORY_LIMIT_PCT = "1%";

    public static final Integer ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE = -1;

    /**
     * Settings Definition
     */

    // This setting controls how much memory should be used to transfer vectors from Java to JNI Layer. The default
    // 1% of the JVM heap
    public static final Setting<ByteSizeValue> KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING = Setting.memorySizeSetting(
        KNN_VECTOR_STREAMING_MEMORY_LIMIT_IN_MB,
        KNN_DEFAULT_VECTOR_STREAMING_MEMORY_LIMIT_PCT,
        Setting.Property.Dynamic,
        Setting.Property.NodeScope
    );

    public static final Setting<String> INDEX_KNN_SPACE_TYPE = Setting.simpleString(
        KNN_SPACE_TYPE,
        INDEX_KNN_DEFAULT_SPACE_TYPE,
        new SpaceTypeValidator(),
        IndexScope,
        Setting.Property.Deprecated
    );

    /**
     * M - the number of bi-directional links created for every new element during construction.
     * Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic
     * dimensionality and/or high recall, while low M work better for datasets with low intrinsic dimensionality and/or low recalls.
     * The parameter also determines the algorithm's memory consumption, which is roughly M * 8-10 bytes per stored element.
     */
    public static final Setting<Integer> INDEX_KNN_ALGO_PARAM_M_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_M,
        INDEX_KNN_DEFAULT_ALGO_PARAM_M,
        2,
        IndexScope,
        Setting.Property.Deprecated
    );

    /**
     *  ef or efSearch - the size of the dynamic list for the nearest neighbors (used during the search).
     *  Higher ef leads to more accurate but slower search. ef cannot be set lower than the number of queried nearest neighbors k.
     *  The value ef can be anything between k and the size of the dataset.
     */
    public static final Setting<Integer> INDEX_KNN_ALGO_PARAM_EF_SEARCH_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_EF_SEARCH,
        INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
        2,
        IndexScope,
        Dynamic
    );

    /**
     * ef_constrution - the parameter has the same meaning as ef, but controls the index_time/index_accuracy.
     * Bigger ef_construction leads to longer construction(more indexing time), but better index quality.
     */
    public static final Setting<Integer> INDEX_KNN_ALGO_PARAM_EF_CONSTRUCTION_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_EF_CONSTRUCTION,
        INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
        2,
        IndexScope,
        Setting.Property.Deprecated
    );

    public static final Setting<Integer> MODEL_INDEX_NUMBER_OF_SHARDS_SETTING = Setting.intSetting(
        MODEL_INDEX_NUMBER_OF_SHARDS,
        1,
        1,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static final Setting<Integer> MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING = Setting.intSetting(
        MODEL_INDEX_NUMBER_OF_REPLICAS,
        1,
        0,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static final Setting<Integer> ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING = Setting.intSetting(
        ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD,
        ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE,
        IndexScope,
        Setting.Property.Dynamic
    );

    public static final Setting<ByteSizeValue> MODEL_CACHE_SIZE_LIMIT_SETTING = new Setting<>(
        MODEL_CACHE_SIZE_LIMIT,
        percentageAsString(KNN_DEFAULT_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE),
        (s) -> {
            ByteSizeValue userDefinedLimit = parseBytesSizeValueOrHeapRatio(s, MODEL_CACHE_SIZE_LIMIT);

            // parseBytesSizeValueOrHeapRatio will make sure that the value entered falls between 0 and 100% of the
            // JVM heap. However, we want the maximum percentage of the heap to be much smaller. So, we add
            // some additional validation here before returning
            ByteSizeValue jvmHeapSize = JvmInfo.jvmInfo().getMem().getHeapMax();
            if ((userDefinedLimit.getKbFrac() / jvmHeapSize.getKbFrac()) > percentageAsFraction(
                KNN_MAX_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE
            )) {
                throw new OpenSearchParseException(
                    "{} ({} KB) cannot exceed {}% of the heap ({} KB).",
                    MODEL_CACHE_SIZE_LIMIT,
                    userDefinedLimit.getKb(),
                    KNN_MAX_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE,
                    jvmHeapSize.getKb()
                );
            }

            return userDefinedLimit;
        },
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    /**
     * This setting identifies KNN index.
     */
    public static final Setting<Boolean> IS_KNN_INDEX_SETTING = Setting.boolSetting(KNN_INDEX, false, IndexScope);

    /**
     * index_thread_quantity - the parameter specifies how many threads the nms library should use to create the graph.
     * By default, the nms library sets this value to NUM_CORES. However, because ES can spawn NUM_CORES threads for
     * indexing, and each indexing thread calls the NMS library to build the graph, which can also spawn NUM_CORES threads,
     * this could lead to NUM_CORES^2 threads running and could lead to 100% CPU utilization. This setting allows users to
     * configure number of threads for graph construction.
     */
    public static final Setting<Integer> KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_INDEX_THREAD_QTY,
        KNN_DEFAULT_ALGO_PARAM_INDEX_THREAD_QTY,
        1,
        INDEX_THREAD_QTY_MAX,
        NodeScope,
        Dynamic
    );

    public static final Setting<Boolean> KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING = Setting.boolSetting(
        KNN_CIRCUIT_BREAKER_TRIGGERED,
        false,
        NodeScope,
        Dynamic
    );

    public static final Setting<Double> KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING = Setting.doubleSetting(
        KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE,
        KNN_DEFAULT_CIRCUIT_BREAKER_UNSET_PERCENTAGE,
        0,
        100,
        NodeScope,
        Dynamic
    );

    public static final Setting<Boolean> KNN_FAISS_AVX2_DISABLED_SETTING = Setting.boolSetting(
        KNN_FAISS_AVX2_DISABLED,
        KNN_DEFAULT_FAISS_AVX2_DISABLED_VALUE,
        NodeScope
    );

    /**
     * Dynamic settings
     */
    public static Map<String, Setting<?>> dynamicCacheSettings = new HashMap<String, Setting<?>>() {
        {
            /**
             * KNN plugin enable/disable setting
             */
            put(KNN_PLUGIN_ENABLED, Setting.boolSetting(KNN_PLUGIN_ENABLED, true, NodeScope, Dynamic));

            /**
             * Weight circuit breaker settings
             */
            put(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, Setting.boolSetting(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, true, NodeScope, Dynamic));
            put(
                KNN_MEMORY_CIRCUIT_BREAKER_LIMIT,
                new Setting<>(
                    KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT,
                    KNNSettings.KNN_DEFAULT_MEMORY_CIRCUIT_BREAKER_LIMIT,
                    (s) -> parseknnMemoryCircuitBreakerValue(s, KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT),
                    NodeScope,
                    Dynamic
                )
            );

            /**
             * Cache expiry time settings
             */
            put(KNN_CACHE_ITEM_EXPIRY_ENABLED, Setting.boolSetting(KNN_CACHE_ITEM_EXPIRY_ENABLED, false, NodeScope, Dynamic));
            put(
                KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES,
                Setting.positiveTimeSetting(KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES, TimeValue.timeValueHours(3), NodeScope, Dynamic)
            );
        }
    };

    private ClusterService clusterService;
    private Client client;

    private KNNSettings() {}

    public static synchronized KNNSettings state() {
        if (INSTANCE == null) {
            INSTANCE = new KNNSettings();
        }
        return INSTANCE;
    }

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

        if (KNN_FAISS_AVX2_DISABLED.equals(key)) {
            return KNN_FAISS_AVX2_DISABLED_SETTING;
        }

        if (KNN_VECTOR_STREAMING_MEMORY_LIMIT_IN_MB.equals(key)) {
            return KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING;
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
            ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING,
            KNN_FAISS_AVX2_DISABLED_SETTING,
            KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING
        );
        return Stream.concat(settings.stream(), dynamicCacheSettings.values().stream()).collect(Collectors.toList());
    }

    public static boolean isKNNPluginEnabled() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_PLUGIN_ENABLED);
    }

    public static boolean isCircuitBreakerTriggered() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED);
    }

    public static ByteSizeValue getCircuitBreakerLimit() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT);
    }

    public static double getCircuitBreakerUnsetPercentage() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE);
    }

    public static boolean isFaissAVX2Disabled() {
        try {
            return KNNSettings.state().getSettingValue(KNNSettings.KNN_FAISS_AVX2_DISABLED);
        } catch (Exception e) {
            // In some UTs we identified that cluster setting is not set properly an leads to NPE. This check will avoid
            // those cases and will still return the default value.
            log.warn(
                "Unable to get setting value {} from cluster settings. Using default value as {}",
                KNN_FAISS_AVX2_DISABLED,
                KNN_DEFAULT_FAISS_AVX2_DISABLED_VALUE,
                e
            );
            return KNN_DEFAULT_FAISS_AVX2_DISABLED_VALUE;
        }
    }

    public static Integer getFilteredExactSearchThreshold(final String indexName) {
        return KNNSettings.state().clusterService.state()
            .getMetadata()
            .index(indexName)
            .getSettings()
            .getAsInt(ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE);
    }

    public void initialize(Client client, ClusterService clusterService) {
        this.client = client;
        this.clusterService = clusterService;
        setSettingsUpdateConsumers();
    }

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

    /**
     * Updates knn.circuit_breaker.triggered setting to true/false
     * @param flag true/false
     */
    public synchronized void updateCircuitBreakerSettings(boolean flag) {
        ClusterUpdateSettingsRequest clusterUpdateSettingsRequest = new ClusterUpdateSettingsRequest();
        Settings circuitBreakerSettings = Settings.builder().put(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED, flag).build();
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

    public static ByteSizeValue getVectorStreamingMemoryLimit() {
        return KNNSettings.state().getSettingValue(KNN_VECTOR_STREAMING_MEMORY_LIMIT_IN_MB);
    }

    /**
     *
     * @param index Name of the index
     * @return efSearch value
     */
    public static int getEfSearchParam(String index) {
        final IndexMetadata indexMetadata = KNNSettings.state().clusterService.state().getMetadata().index(index);
        return indexMetadata.getSettings()
            .getAsInt(
                KNNSettings.KNN_ALGO_PARAM_EF_SEARCH,
                IndexHyperParametersUtil.getHNSWEFSearchValue(indexMetadata.getCreationVersion())
            );
    }

    public void setClusterService(ClusterService clusterService) {
        this.clusterService = clusterService;
    }

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

    private static String percentageAsString(Integer percentage) {
        return percentage + "%";
    }

    private static Double percentageAsFraction(Integer percentage) {
        return percentage / 100.0;
    }
}
