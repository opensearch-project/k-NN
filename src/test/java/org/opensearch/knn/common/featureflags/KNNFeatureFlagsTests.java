/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.featureflags;

import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import static org.mockito.Mockito.when;
import java.util.List;

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

    public void testIsPrefetchEnabled() {
        when(clusterSettings.get(KNN_PREFETCH_ENABLED_SETTING)).thenReturn(true);
        assertTrue(isPrefetchEnabled());
        when(clusterSettings.get(KNN_PREFETCH_ENABLED_SETTING)).thenReturn(false);
        assertFalse(isPrefetchEnabled());
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
