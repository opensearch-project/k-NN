/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.index.shard.IndexSettingProvider;

public class KNNIndexSettingProvider implements IndexSettingProvider {
    public static final ByteSizeValue KNN_DEFAULT_FLOOR_SEGMENT_VALUE = new ByteSizeValue(16, ByteSizeUnit.MB);

    private static boolean isKNNIndex(Settings settings) {
        return settings.hasValue("index.knn") && settings.getAsBoolean("index.knn", true);
    }

    /**
     * Returns additional index settings for k-NN index. In particular, we set the index.merge.policy.floor_segment = 16MB.
     * This change is in line with Lucene 10.2 default and will lead to fewer segments (more merges), improving search performance.
     */
    @Override
    public Settings getAdditionalIndexSettings(String indexName, boolean isDataStreamIndex, Settings templateAndRequestSettings) {
        if (isKNNIndex(templateAndRequestSettings)) {
            return Settings.builder().put("index.merge.policy.floor_segment", KNN_DEFAULT_FLOOR_SEGMENT_VALUE).build();
        } else {
            return Settings.EMPTY;
        }
    }
}
