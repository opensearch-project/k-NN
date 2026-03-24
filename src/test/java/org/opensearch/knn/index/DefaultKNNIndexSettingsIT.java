/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.index.TieredMergePolicyProvider;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.common.settings.Settings;

import static org.opensearch.knn.index.KNNSettings.KNN_DERIVED_SOURCE_ENABLED;

public class DefaultKNNIndexSettingsIT extends KNNRestTestCase {

    public void testKnnIndexSettings_whenisKNN_thenOverride() throws Exception {
        String indexName = randomLowerCaseString();
        String fieldName = randomLowerCaseString();
        boolean isKnnIndex = randomBoolean();

        Settings indexSettings = Settings.builder().put("index.knn", isKnnIndex).build();

        if (isKnnIndex) {
            String mapping = createKnnIndexMapping(fieldName, 128);
            createKnnIndex(indexName, indexSettings, mapping);
        } else {
            createIndex(indexName, indexSettings);
        }

        // Verify settings based on knn flag
        String derivedSourceEnabled = getIndexSettingByName(indexName, KNN_DERIVED_SOURCE_ENABLED);
        assertEquals(!isKnnIndex ? null : "true", derivedSourceEnabled);

        String maxMergeAtOnce = getIndexSettingByName(
            indexName,
            TieredMergePolicyProvider.INDEX_MERGE_POLICY_MAX_MERGE_AT_ONCE_SETTING.getKey(),
            !isKnnIndex
        );
        String floorSegment = getIndexSettingByName(
            indexName,
            TieredMergePolicyProvider.INDEX_MERGE_POLICY_FLOOR_SEGMENT_SETTING.getKey(),
            !isKnnIndex
        );

        if (isKnnIndex) {
            assertEquals("10", maxMergeAtOnce);
            assertEquals("2mb", floorSegment);
        } else {
            assertNotEquals("10", maxMergeAtOnce);
            assertNotEquals("2mb", floorSegment);
        }

        deleteKNNIndex(indexName);
    }
}
