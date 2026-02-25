/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.knn.common.featureflags;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import lombok.experimental.UtilityClass;
import org.opensearch.common.Booleans;
import org.opensearch.common.settings.Setting;
import org.opensearch.knn.index.KNNSettings;

import java.util.List;

import static org.opensearch.common.settings.Setting.Property.Dynamic;
import static org.opensearch.common.settings.Setting.Property.NodeScope;

/**
 * Class to manage KNN feature flags
 */
@UtilityClass
public class KNNFeatureFlags {

    // Feature flags
    private static final String KNN_FORCE_EVICT_CACHE_ENABLED = "knn.feature.cache.force_evict.enabled";

    @VisibleForTesting
    public static final Setting<Boolean> KNN_FORCE_EVICT_CACHE_ENABLED_SETTING = Setting.boolSetting(
        KNN_FORCE_EVICT_CACHE_ENABLED,
        false,
        NodeScope,
        Dynamic
    );

    public static List<Setting<?>> getFeatureFlags() {
        return ImmutableList.of(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING);
    }

    /**
     * Checks if force evict for cache is enabled by executing a check against cluster settings
     * @return true if force evict setting is set to true
     */
    public static boolean isForceEvictCacheEnabled() {
        return Booleans.parseBoolean(KNNSettings.state().getSettingValue(KNN_FORCE_EVICT_CACHE_ENABLED).toString(), false);
    }
}
