/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.featureflags;

import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import static org.mockito.Mockito.when;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.KNN_FORCE_EVICT_CACHE_ENABLED_SETTING;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.KNN_PREFETCH_ENABLED_SETTING;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.isForceEvictCacheEnabled;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.isPrefetchEnabled;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.getFeatureFlags;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.getFeatureFlagsWhichRebuildsCache;

public class KNNFeatureFlagsTests extends KNNTestCase {

    @Mock
    ClusterSettings clusterSettings;

    public void setUp() throws Exception {
        super.setUp();
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
    }

    public void testIsForceEvictCacheEnabled() throws Exception {
        when(clusterSettings.get(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING)).thenReturn(false);
        assertFalse(isForceEvictCacheEnabled());
        when(clusterSettings.get(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING)).thenReturn(true);
        assertTrue(isForceEvictCacheEnabled());
    }

    public void testIsPrefetchEnabled_whenNotInitialized_thenDefault() {
        assertEquals(KNN_PREFETCH_ENABLED_SETTING.getDefault(Settings.EMPTY), isPrefetchEnabled());
        assertTrue(isPrefetchEnabled());
    }

    public void testIsPrefetchEnabled_whenSettingUnset_thenDefault() {
        try {
            KNNFeatureFlags.initialize(newClusterSettings(Settings.EMPTY));
            assertTrue(isPrefetchEnabled());
        } finally {
            resetPrefetchEnabled();
        }
    }

    public void testIsPrefetchEnabled_whenSetInNodeSettings_thenInitialValueRespected() {
        try {
            KNNFeatureFlags.initialize(newClusterSettings(Settings.builder().put(KNN_PREFETCH_ENABLED_SETTING.getKey(), false).build()));
            assertFalse(isPrefetchEnabled());
        } finally {
            resetPrefetchEnabled();
        }
    }

    public void testIsPrefetchEnabled_whenClusterSettingUpdated_thenCachedValueUpdated() {
        try {
            final ClusterSettings realClusterSettings = newClusterSettings(Settings.EMPTY);
            KNNFeatureFlags.initialize(realClusterSettings);
            assertTrue(isPrefetchEnabled());

            realClusterSettings.applySettings(Settings.builder().put(KNN_PREFETCH_ENABLED_SETTING.getKey(), false).build());
            assertFalse(isPrefetchEnabled());

            realClusterSettings.applySettings(Settings.builder().put(KNN_PREFETCH_ENABLED_SETTING.getKey(), true).build());
            assertTrue(isPrefetchEnabled());

            realClusterSettings.applySettings(Settings.builder().put(KNN_PREFETCH_ENABLED_SETTING.getKey(), false).build());
            assertFalse(isPrefetchEnabled());

            // Removing the setting falls back to the default
            realClusterSettings.applySettings(Settings.EMPTY);
            assertTrue(isPrefetchEnabled());
        } finally {
            resetPrefetchEnabled();
        }
    }

    private static ClusterSettings newClusterSettings(final Settings nodeSettings) {
        return new ClusterSettings(nodeSettings, Set.of(KNN_PREFETCH_ENABLED_SETTING));
    }

    private static void resetPrefetchEnabled() {
        KNNFeatureFlags.initialize(newClusterSettings(Settings.EMPTY));
    }

    public void testGetFeatureFlags() {
        List<Setting<?>> flags = getFeatureFlags();
        assertEquals(2, flags.size());
        assertTrue(flags.contains(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING));
        assertTrue(flags.contains(KNN_PREFETCH_ENABLED_SETTING));
    }

    public void testGetFeatureFlagsWhichRebuildsCache() {
        List<Setting<?>> flags = getFeatureFlagsWhichRebuildsCache();
        assertEquals(1, flags.size());
        assertTrue(flags.contains(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING));
        assertFalse(flags.contains(KNN_PREFETCH_ENABLED_SETTING));
    }
}
