/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.annotations.VisibleForTesting;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.util.Strings;
import org.opensearch.Version;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Locale;

import static org.opensearch.knn.index.engine.KNNEngine.DEPRECATED_ENGINES;

/**
 * Figures out what {@link KNNEngine} to use based on configuration details
 */
public final class EngineResolver {

    private static Logger logger = LogManager.getLogger(EngineResolver.class);
    public static final EngineResolver INSTANCE = new EngineResolver();

    private EngineResolver() {}

    @VisibleForTesting
    KNNEngine resolveEngine(
        KNNMethodConfigContext knnMethodConfigContext,
        KNNMethodContext knnMethodContext,
        String topLevelString,
        boolean requiresTraining
    ) {
        return logAndReturnEngine(
            resolveKNNEngine(knnMethodConfigContext, knnMethodContext, topLevelString, requiresTraining, Version.CURRENT)
        );
    }

    /**
     * Resolves engine from configuration details. It is guaranteed not to return null.
     * When engine is not in either method and top level, DEFAULT will be returned.
     *
     * @param knnMethodConfigContext configuration context
     * @param knnMethodContext KNNMethodContext
     * @param topLevelEngineString Alternative top-level engine
     * @return {@link SpaceType} for the method
     */
    public KNNEngine resolveEngine(
        KNNMethodConfigContext knnMethodConfigContext,
        KNNMethodContext knnMethodContext,
        String topLevelEngineString,
        boolean requiresTraining,
        Version version
    ) {
        return logAndReturnEngine(
            resolveKNNEngine(knnMethodConfigContext, knnMethodContext, topLevelEngineString, requiresTraining, version)
        );
    }

    /**
     * Based on the provided {@link Mode} and {@link CompressionLevel}, resolve to a {@link KNNEngine}.
     *
     * @param knnMethodConfigContext configuration context
     * @param knnMethodContext KNNMethodContext
     * @param topLevelEngineString Alternative top-level engine
     * @param requiresTraining whether config requires training
     * @param version opensearch index version
     * @return {@link KNNEngine}
     */
    private KNNEngine resolveKNNEngine(
        KNNMethodConfigContext knnMethodConfigContext,
        KNNMethodContext knnMethodContext,
        String topLevelEngineString,
        boolean requiresTraining,
        Version version
    ) {
        KNNEngine methodEngine = getEngineTypeFromMethodContext(knnMethodContext);
        KNNEngine topLevelEngine = getEngineFromString(topLevelEngineString);

        // user configured method engine
        if (isEngineConfigured(topLevelEngine) == false && hasUserConfiguredEngine(knnMethodContext)) {
            return methodEngine;
        }

        // user configured top level engine
        if (hasUserConfiguredEngine(knnMethodContext) == false && isEngineConfigured(topLevelEngine) != false) {
            return topLevelEngine;
        }

        if (isEngineConfigured(topLevelEngine) && topLevelEngine == methodEngine && hasUserConfiguredEngine(knnMethodContext)) {
            // both engines are same
            return topLevelEngine;
        } else if (isEngineConfigured(topLevelEngine) && topLevelEngine != methodEngine && hasUserConfiguredEngine(knnMethodContext)) {
            // engines are different
            throw new MapperParsingException(
                String.format(
                    Locale.ROOT,
                    "Cannot specify conflicting engines: \"[%s]\" \"[%s]\"",
                    methodEngine.getName(),
                    topLevelEngine.getName()
                )
            );
        }

        // Handle training case
        if (requiresTraining) {
            // Faiss is the only engine that supports training, so we default to faiss here for now
            return KNNEngine.FAISS;
        }

        Mode mode = knnMethodConfigContext.getMode();
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();

        if (compressionLevel == CompressionLevel.x4) {
            // Lucene is only engine that supports 4x - so we have to default to it here.
            return KNNEngine.LUCENE;
        }
        if (CompressionLevel.isConfigured(compressionLevel) == false || compressionLevel == CompressionLevel.x1) {
            // For 1x or no compression, we need to default to faiss if mode is provided and use nmslib otherwise based on version check
            return resolveEngineForX1OrNoCompression(mode, version);
        }

        // If both mode and compression are not specified, we can just default
        if (Mode.isConfigured(mode) == false && CompressionLevel.isConfigured(compressionLevel) == false) {
            return KNNEngine.DEFAULT;
        }
        return KNNEngine.FAISS;
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

    private KNNEngine logAndReturnEngine(KNNEngine knnEngine) {
        if (DEPRECATED_ENGINES.contains(knnEngine)) {
            logger.warn("[Deprecation] {} engine is deprecated and will be removed in a future release.", knnEngine);
        }
        return knnEngine;
    }

    private boolean isEngineConfigured(final KNNEngine knnEngine) {
        return knnEngine != null && knnEngine != KNNEngine.UNDEFINED;
    }

    private KNNEngine getEngineTypeFromMethodContext(final KNNMethodContext knnMethodContext) {
        if (knnMethodContext == null) {
            return KNNEngine.UNDEFINED;
        }

        return knnMethodContext.getKnnEngine();
    }

    private KNNEngine getEngineFromString(final String knnEngineString) {
        if (Strings.isEmpty(knnEngineString)) {
            return KNNEngine.UNDEFINED;
        }

        return KNNEngine.getEngine(knnEngineString);
    }
}
