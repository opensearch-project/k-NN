/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.List;

public class KNNPluginTests extends OpenSearchTestCase {

    public void testKNNPlugin_additionalIndexProviderSettings() throws IOException {
        try (KNNPlugin knnPlugin = new KNNPlugin()) {
            Settings additionalSettings = knnPlugin.getAdditionalIndexSettingProviders()
                .iterator()
                .next()
                .getAdditionalIndexSettings("index", false, Settings.builder().put(KNNSettings.KNN_INDEX, Boolean.TRUE).build());

            Settings settings = Settings.builder().putList("index.store.preload", List.of("vec", "vex")).build();
            assertEquals(settings, additionalSettings);

            additionalSettings = knnPlugin.getAdditionalIndexSettingProviders()
                .iterator()
                .next()
                .getAdditionalIndexSettings("index", false, Settings.builder().build());

            assertEquals(Settings.EMPTY, additionalSettings);
        }
    }
}
