/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexModule;
import org.opensearch.index.shard.IndexSettingProvider;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Collection;
import java.util.List;

public class KNNPluginTests extends KNNTestCase {

    private static final String INDEX_NAME = "test_index";

    public void testMMapFileExtensionsForHybridFs() {
        final KNNPlugin knnPlugin = new KNNPlugin();

        final Collection<IndexSettingProvider> additionalIndexSettingProviders = knnPlugin.getAdditionalIndexSettingProviders();

        assertEquals(1, additionalIndexSettingProviders.size());

        final IndexSettingProvider indexSettingProvider = additionalIndexSettingProviders.iterator().next();
        // settings for knn enabled index
        final Settings knnIndexSettings = indexSettingProvider.getAdditionalIndexSettings(INDEX_NAME, false, getKnnDefaultIndexSettings());
        final List<String> mmapFileExtensionsForHybridFsKnnIndex = knnIndexSettings.getAsList(
            IndexModule.INDEX_STORE_HYBRID_MMAP_EXTENSIONS.getKey()
        );

        assertNotNull(mmapFileExtensionsForHybridFsKnnIndex);
        assertFalse(mmapFileExtensionsForHybridFsKnnIndex.isEmpty());

        for (KNNEngine engine : KNNEngine.values()) {
            assertTrue(mmapFileExtensionsForHybridFsKnnIndex.containsAll(engine.mmapFileExtensions()));
        }

        // settings for index without knn
        final Settings nonKnnIndexSettings = indexSettingProvider.getAdditionalIndexSettings(INDEX_NAME, false, getNonKnnIndexSettings());
        final List<String> mmapFileExtensionsForHybridFsNonKnnIndex = nonKnnIndexSettings.getAsList(
            IndexModule.INDEX_STORE_HYBRID_MMAP_EXTENSIONS.getKey()
        );

        assertNotNull(mmapFileExtensionsForHybridFsNonKnnIndex);
        assertTrue(mmapFileExtensionsForHybridFsNonKnnIndex.isEmpty());
    }

    private Settings getKnnDefaultIndexSettings() {
        return Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", true).build();
    }

    private Settings getNonKnnIndexSettings() {
        return Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).build();
    }
}
