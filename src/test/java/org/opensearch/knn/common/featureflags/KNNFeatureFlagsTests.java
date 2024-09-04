/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common.featureflags;

import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.KNN_FORCE_EVICT_CACHE_ENABLED_SETTING;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.isForceEvictCacheEnabled;

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
}
