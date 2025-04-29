/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import static org.opensearch.knn.index.engine.KNNEngine.DEPRECATED_ENGINES;

/**
 * Figures out what {@link KNNEngine} to use based on configuration details
 */
public final class EngineResolver {

    private static Logger logger = LogManager.getLogger(EngineResolver.class);
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
        // User configuration gets precedence
        if (knnMethodContext != null && knnMethodContext.isEngineConfigured()) {
            return logAndReturnEngine(knnMethodContext.getKnnEngine());
        }

        // Faiss is the only engine that supports training, so we default to faiss here for now
        if (requiresTraining) {
            return logAndReturnEngine(KNNEngine.FAISS);
        }

        Mode mode = knnMethodConfigContext.getMode();
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        // If both mode and compression are not specified, we can just default
        if (Mode.isConfigured(mode) == false && CompressionLevel.isConfigured(compressionLevel) == false) {
            return logAndReturnEngine(KNNEngine.DEFAULT);
        }

        // For 1x, we need to default to faiss if mode is provided and use nmslib otherwise
        if (CompressionLevel.isConfigured(compressionLevel) == false || compressionLevel == CompressionLevel.x1) {
            return logAndReturnEngine(mode == Mode.ON_DISK ? KNNEngine.FAISS : KNNEngine.NMSLIB);
        }

        // Lucene is only engine that supports 4x - so we have to default to it here.
        if (compressionLevel == CompressionLevel.x4) {
            return logAndReturnEngine(KNNEngine.LUCENE);
        }

        return logAndReturnEngine(KNNEngine.FAISS);
    }

    private KNNEngine logAndReturnEngine(KNNEngine knnEngine) {
        if (DEPRECATED_ENGINES.contains(knnEngine)) {
            logger.warn("[Deprecation] {} engine is deprecated and will be removed in a future release.", knnEngine);
        }
        return knnEngine;
    }
}
