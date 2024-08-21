/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.knn.common.featureflags;

import com.google.common.annotations.VisibleForTesting;
import lombok.experimental.UtilityClass;
import org.opensearch.common.settings.Setting;
import org.opensearch.knn.index.KNNSettings;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.opensearch.common.settings.Setting.Property.Dynamic;
import static org.opensearch.common.settings.Setting.Property.NodeScope;

/**
 * Class to manage KNN feature flags
 */
@UtilityClass
public class KNNFeatureFlags {

    // Feature flags
    private static final String KNN_LAUNCH_QUERY_REWRITE_ENABLED = "knn.feature.query.rewrite.enabled";
    private static final boolean KNN_LAUNCH_QUERY_REWRITE_ENABLED_DEFAULT = false;

    /**
     * TODO: This setting is only added to ensure that main branch of k_NN plugin doesn't break till other parts of the
     * code is getting ready. Will remove this setting once all changes related to integration of KNNVectorsFormat is added
     * for native engines.
     */
    public static final String KNN_USE_LUCENE_VECTOR_FORMAT_ENABLED = "knn.use.format.enabled";

    @VisibleForTesting
    public static final Setting<Boolean> KNN_LAUNCH_QUERY_REWRITE_ENABLED_SETTING = Setting.boolSetting(
        KNN_LAUNCH_QUERY_REWRITE_ENABLED,
        KNN_LAUNCH_QUERY_REWRITE_ENABLED_DEFAULT,
        NodeScope,
        Dynamic
    );

    /**
     * TODO: This setting is only added to ensure that main branch of k_NN plugin doesn't break till other parts of the
     * code is getting ready. Will remove this setting once all changes related to integration of KNNVectorsFormat is added
     * for native engines.
     */
    public static final Setting<Boolean> KNN_USE_LUCENE_VECTOR_FORMAT_ENABLED_SETTING = Setting.boolSetting(
        KNN_USE_LUCENE_VECTOR_FORMAT_ENABLED,
        true,
        NodeScope,
        Dynamic
    );

    /**
     * TODO: This setting is only added to ensure that main branch of k_NN plugin doesn't break till other parts of the
     * code is getting ready. Will remove this setting once all changes related to integration of KNNVectorsFormat is added
     * for native engines.
     */
    public static boolean getIsLuceneVectorFormatEnabled() {
        return KNNSettings.state().getSettingValue(KNN_USE_LUCENE_VECTOR_FORMAT_ENABLED);
    }

    public static List<Setting<?>> getFeatureFlags() {
        return Stream.of(KNN_LAUNCH_QUERY_REWRITE_ENABLED_SETTING, KNN_USE_LUCENE_VECTOR_FORMAT_ENABLED_SETTING)
            .collect(Collectors.toUnmodifiableList());
    }

    public static boolean isKnnQueryRewriteEnabled() {
        return Boolean.parseBoolean(KNNSettings.state().getSettingValue(KNN_LAUNCH_QUERY_REWRITE_ENABLED).toString());
    }
}
