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
    public static final ByteSizeValue CHANGED_KNN_VALUE = new ByteSizeValue(16, ByteSizeUnit.MB);

    @Override
    public Settings getAdditionalIndexSettings(String indexName, boolean isDataStreamIndex, Settings templateAndRequestSettings) {
        if (isKNNIndex(templateAndRequestSettings)) {
            return Settings.builder().put("index.merge.policy.floor_segment", CHANGED_KNN_VALUE).build();
        } else {
            return Settings.EMPTY;
        }
    }

    private boolean isKNNIndex(Settings settings) {
        return settings.hasValue("index.knn") && settings.getAsBoolean("index.knn", true);
    }
}
