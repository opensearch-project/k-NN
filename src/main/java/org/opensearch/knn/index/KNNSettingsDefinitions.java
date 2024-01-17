/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.common.settings.Setting;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;
import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED_SETTING;
import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT;
import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT_SETTING;
import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.KNN_CACHE_ITEM_EXPIRY_ENABLED;
import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.KNN_CACHE_ITEM_EXPIRY_ENABLED_SETTING;
import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES;
import static org.opensearch.knn.index.memory.NativeMemoryCacheManager.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES_SETTING;
import static org.opensearch.knn.plugin.KNNPlugin.KNN_PLUGIN_ENABLED;
import static org.opensearch.knn.plugin.KNNPlugin.KNN_PLUGIN_ENABLED_SETTING;

/**
 * This class simply defines all of the settings and their names for k-NN
 */
public class KNNSettingsDefinitions {
    // TODO: Next steps:
    // 1. Move around all of the settings to the best place they fit
    // 2. Refactor dynamiccachesettings better
    // 2. Move utility functions to best locations
    // 3. Move KNNSettings to KNNCLusterUtil
    // 4. Solidify testing logic

    /**
     * Dynamic settings
     */
    public static Map<String, Setting<?>> dynamicCacheSettings = new HashMap<String, Setting<?>>() {
        {
            put(KNN_PLUGIN_ENABLED, KNN_PLUGIN_ENABLED_SETTING);

            put(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, KNN_MEMORY_CIRCUIT_BREAKER_ENABLED_SETTING);
            put(KNN_MEMORY_CIRCUIT_BREAKER_LIMIT, KNN_MEMORY_CIRCUIT_BREAKER_LIMIT_SETTING);

            put(KNN_CACHE_ITEM_EXPIRY_ENABLED, KNN_CACHE_ITEM_EXPIRY_ENABLED_SETTING);
            put(KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES, KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES_SETTING);
        }
    };
}
