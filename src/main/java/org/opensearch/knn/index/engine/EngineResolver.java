/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

/**
 * Figures out what {@link KNNEngine} to use based on configuration details
 */
public final class EngineResolver {

    public static final EngineResolver INSTANCE = new EngineResolver();

    private EngineResolver() {}

    /**
     * Based on the provided {@link Mode} and {@link CompressionLevel}, resolve to a {@link KNNEngine}.
     *
     * @param knnMethodConfigContext configuration context
     * @param knnMethodContext KNNMethodContext
     * @param requiresTraining whether config requires training
     * @return {@link KNNEngine}
     */
    public KNNEngine resolveEngine(
        KNNMethodConfigContext knnMethodConfigContext,
        KNNMethodContext knnMethodContext,
        boolean requiresTraining
    ) {
        return resolveEngine(knnMethodConfigContext, knnMethodContext, requiresTraining, Version.CURRENT);
    }

    /**
     * Based on the provided {@link Mode} and {@link CompressionLevel}, resolve to a {@link KNNEngine}.
     *
     * @param knnMethodConfigContext configuration context
     * @param knnMethodContext KNNMethodContext
     * @param requiresTraining whether config requires training
     * @param version opensearch index version
     * @return {@link KNNEngine}
     */
    public KNNEngine resolveEngine(
        KNNMethodConfigContext knnMethodConfigContext,
        KNNMethodContext knnMethodContext,
        boolean requiresTraining,
        Version version
    ) {
        // Check user configuration first
        if (hasUserConfiguredEngine(knnMethodContext)) {
            return knnMethodContext.getKnnEngine();
        }

        // Handle training case
        if (requiresTraining) {
            // Faiss is the only engine that supports training, so we default to faiss here for now
            return KNNEngine.FAISS;
        }

        Mode mode = knnMethodConfigContext.getMode();
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();

        // If both mode and compression are not specified, we can just default
        if (Mode.isConfigured(mode) == false && CompressionLevel.isConfigured(compressionLevel) == false) {
            return KNNEngine.DEFAULT;
        }

        return switch (compressionLevel) {
            // Lucene is only engine that supports 4x - so we have to default to it here.
            case x4 -> KNNEngine.LUCENE;
            // For 1x or no compression, we need to default to faiss if mode is provided and use nmslib otherwise based on version check
            case x1, CompressionLevel.NOT_CONFIGURED -> resolveEngineForX1OrNoCompression(mode, version);
            default -> KNNEngine.FAISS;
        };
    }

    private boolean hasUserConfiguredEngine(KNNMethodContext knnMethodContext) {
        return knnMethodContext != null && knnMethodContext.isEngineConfigured();
    }

    private KNNEngine resolveEngineForX1OrNoCompression(Mode mode, Version version) {
        if (version != null && version.onOrAfter(Version.V_2_19_0)) {
            return KNNEngine.FAISS;
        }
        return mode == Mode.ON_DISK ? KNNEngine.FAISS : KNNEngine.NMSLIB;
    }

}
