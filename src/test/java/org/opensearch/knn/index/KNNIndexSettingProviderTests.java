/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.plugin.KNNIndexSettingProvider;

import static org.opensearch.index.TieredMergePolicyProvider.INDEX_MERGE_POLICY_FLOOR_SEGMENT_SETTING;
import static org.opensearch.knn.index.KNNSettings.IS_KNN_INDEX_SETTING;

public class KNNIndexSettingProviderTests extends KNNSingleNodeTestCase {

    private final KNNIndexSettingProvider provider = new KNNIndexSettingProvider();
    private final String testIndexName = "test-index";

    public void testGetAdditionalIndexSettings_knnIndexEnabled() {
        Settings templateSettings = Settings.builder()
                .put(IS_KNN_INDEX_SETTING.getKey(), true)
                .build();

        Settings additionalSettings = provider.getAdditionalIndexSettings(testIndexName, false, templateSettings);

        assertNotNull(additionalSettings);
        assertEquals(
                KNNIndexSettingProvider.KNN_DEFAULT_FLOOR_SEGMENT_VALUE,
                additionalSettings.getAsBytesSize(INDEX_MERGE_POLICY_FLOOR_SEGMENT_SETTING.getKey(), ByteSizeValue.ZERO)
        );
    }

    public void testGetAdditionalIndexSettings_knnIndexDisabled() {
        Settings templateSettings = Settings.builder()
                .put(IS_KNN_INDEX_SETTING.getKey(), false)
                .build();

        Settings additionalSettings = provider.getAdditionalIndexSettings(testIndexName, false, templateSettings);

        assertNotNull(additionalSettings);
        assertTrue(additionalSettings.isEmpty());
    }

    public void testGetAdditionalIndexSettings_noKnnSetting() {
        Settings templateSettings = Settings.builder().build();

        Settings additionalSettings = provider.getAdditionalIndexSettings(testIndexName, false, templateSettings);

        assertNotNull(additionalSettings);
        assertTrue(additionalSettings.isEmpty());
    }

    public void testGetAdditionalIndexSettings_dataStreamIndex() {
        Settings templateSettings = Settings.builder()
                .put(IS_KNN_INDEX_SETTING.getKey(), true)
                .build();

        Settings additionalSettings = provider.getAdditionalIndexSettings(testIndexName, true, templateSettings);

        assertNotNull(additionalSettings);
        assertEquals(
                KNNIndexSettingProvider.KNN_DEFAULT_FLOOR_SEGMENT_VALUE,
                additionalSettings.getAsBytesSize(INDEX_MERGE_POLICY_FLOOR_SEGMENT_SETTING.getKey(), ByteSizeValue.ZERO)
        );
    }
}