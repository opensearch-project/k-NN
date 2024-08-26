/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.index.engine.config.CompressionConfig;
import org.opensearch.knn.index.engine.config.WorkloadModeConfig;
import org.opensearch.knn.index.util.IndexUtil;

import static org.opensearch.knn.common.KNNConstants.MINIMAL_MODE_AND_COMPRESSION_FEATURE;
import static org.opensearch.knn.index.engine.KNNEngine.FAISS;
import static org.opensearch.knn.index.engine.KNNEngine.NMSLIB;

/**
 * Utility xlass used to resolve the engine for a k-NN method config context
 */
public class KNNEngineResolver {

    public static final KNNEngine LEGACY_DEFAULT = NMSLIB;

    /**
     * Resolves the engine, given the context
     *
     * @param knnMethodConfigContext context to use for resolution
     * @return engine to use for the knn method
     */
    public static KNNEngine resolveKNNEngine(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext == null) {
            return getDefault(null);
        }

        KNNMethodContext knnMethodContext = knnMethodConfigContext.getKnnMethodContext();
        if (knnMethodContext == null) {
            return getDefault(knnMethodConfigContext);
        }

        return knnMethodContext.getKnnEngine().orElse(getDefault(knnMethodConfigContext));
    }

    private static KNNEngine getDefault(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext == null) {
            return NMSLIB;
        }

        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(
            knnMethodConfigContext.getVersionCreated(),
            MINIMAL_MODE_AND_COMPRESSION_FEATURE
        ) == false) {
            return LEGACY_DEFAULT;
        }

        if (isWorkloadModeNotConfiguredOrDefault(knnMethodConfigContext) && isCompressionNotConfiguredOrDefault(knnMethodConfigContext)) {
            return NMSLIB;
        }

        return FAISS;
    }

    private static boolean isWorkloadModeNotConfiguredOrDefault(KNNMethodConfigContext knnMethodConfigContext) {
        return knnMethodConfigContext.getWorkloadModeConfig() == WorkloadModeConfig.NOT_CONFIGURED
            || knnMethodConfigContext.getWorkloadModeConfig() == WorkloadModeConfig.DEFAULT;
    }

    private static boolean isCompressionNotConfiguredOrDefault(KNNMethodConfigContext knnMethodConfigContext) {
        return knnMethodConfigContext.getCompressionConfig() == CompressionConfig.NOT_CONFIGURED
            || knnMethodConfigContext.getCompressionConfig() == CompressionConfig.DEFAULT;
    }
}
