/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;

import static org.opensearch.knn.index.KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE;

public class KNNCommonSettingsBuilder {
    protected Settings.Builder settings;
    protected final int DEFAULT_MULTI_SHARD = 8;

    public static KNNCommonSettingsBuilder defaultSettings() {
        KNNCommonSettingsBuilder knnCommonSettingsBuilder = new KNNCommonSettingsBuilder();
        knnCommonSettingsBuilder.settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0);
        return knnCommonSettingsBuilder;
    }

    public KNNCommonSettingsBuilder multiShard() {
        settings.put("number_of_shards", DEFAULT_MULTI_SHARD);
        return this;
    }

    public KNNCommonSettingsBuilder multiShard(int shards) {
        settings.put("number_of_shards", shards);
        return this;
    }

    public KNNCommonSettingsBuilder memOptSearch() {
        settings.put(MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true);
        return this;
    }

    public Settings build() {
        return settings.build();
    }
}
