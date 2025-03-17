/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.index.shard.IndexSettingProvider;

public class KNNIndexSettingProvider implements IndexSettingProvider {
    public static final ByteSizeValue CHANGED_KNN_VALUE = new ByteSizeValue(16, ByteSizeUnit.MB);
    private static Logger logger = LogManager.getLogger(KNNIndexSettingProvider.class);
    @Override
    public Settings getAdditionalIndexSettings(String indexName, boolean isDataStreamIndex, Settings templateAndRequestSettings) {
        // skipping isKNN logic and will verify with unit tests that it's ok.
        // TODO figure out how to include the other settings, and don't change other plugin behavior.
//            return Settings.builder()
//                    .put("index.merge.policy.floor_segment", "16mb")
//                    .build();
//        }

        System.out.println("template and request settings: ");
        System.out.println(templateAndRequestSettings);
        logger.info("hereherherhe\n\n\n\n\n\n\n");
        logger.info(templateAndRequestSettings);
        logger.info("we're expecting isKnNindex", isKNNIndex(templateAndRequestSettings));
        if (isKNNIndex(templateAndRequestSettings)) {
            return Settings.builder().put("index.merge.policy.floor_segment", CHANGED_KNN_VALUE).build();
        } else {
            return Settings.EMPTY;
        }

//        return Settings.EMPTY;
    }

    private boolean isKNNIndex(Settings settings) {
        // check if knn-specific settings are present
        return settings.hasValue("index.knn") && settings.getAsBoolean("index.knn", true);
        //[2025-03-17T16:08:43,029][INFO ][o.o.k.p.KNNIndexSettingProvider] [integTest-0] {"index.knn":"true","index.knn.algo_param.ef_search":"100"}
    }
}

