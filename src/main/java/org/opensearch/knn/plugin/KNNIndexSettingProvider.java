/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.index.shard.IndexSettingProvider;

import static org.opensearch.index.TieredMergePolicyProvider.INDEX_MERGE_POLICY_FLOOR_SEGMENT_SETTING;
import static org.opensearch.knn.index.KNNSettings.IS_KNN_INDEX_SETTING;

public class KNNIndexSettingProvider implements IndexSettingProvider {
    public static final ByteSizeValue KNN_DEFAULT_FLOOR_SEGMENT_VALUE = new ByteSizeValue(16, ByteSizeUnit.MB);

    /**
     * Returns additional index settings for k-NN index. In particular, we set the index.merge.policy.floor_segment = 16MB.
     * This change is in line with Lucene 10.2 default and will lead to fewer segments (more merges), improving search performance.
     */
    @Override
    public Settings getAdditionalIndexSettings(String indexName, boolean isDataStreamIndex, Settings templateAndRequestSettings) {
        if (IS_KNN_INDEX_SETTING.get(templateAndRequestSettings)) {
            return Settings.builder().put(INDEX_MERGE_POLICY_FLOOR_SEGMENT_SETTING.getKey(), KNN_DEFAULT_FLOOR_SEGMENT_VALUE).build();
        }
        return Settings.EMPTY;
    }
}
