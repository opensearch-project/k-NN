/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_DEFAULT;

/**
 * Class contains the logic to make parameter resolutions based on the {@link Mode} and {@link CompressionLevel}.
 */
public final class ModeBasedResolver {

    public static final ModeBasedResolver INSTANCE = new ModeBasedResolver();

    private static final CompressionLevel DEFAULT_COMPRESSION_FOR_MODE_ON_DISK = CompressionLevel.x32;
    private static final CompressionLevel DEFAULT_COMPRESSION_FOR_MODE_IN_MEMORY = CompressionLevel.x1;
    public final static Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x2,
        CompressionLevel.x8,
        CompressionLevel.x16,
        CompressionLevel.x32
    );

    private ModeBasedResolver() {}

    /**
     * Based on the provided {@link Mode} and {@link CompressionLevel}, resolve to a {@link KNNMethodContext}
     *
     * @param mode {@link Mode}
     * @param compressionLevel {@link CompressionLevel}
     * @param requiresTraining whether config requires trianing
     * @return {@link KNNMethodContext}
     */
    public KNNMethodContext resolveKNNMethodContext(
        Mode mode,
        CompressionLevel compressionLevel,
        boolean requiresTraining,
        SpaceType spaceType
    ) {
        if (requiresTraining) {
            return resolveWithTraining(mode, compressionLevel, spaceType);
        }
        return resolveWithoutTraining(mode, compressionLevel, spaceType);
    }

    private KNNMethodContext resolveWithoutTraining(Mode mode, CompressionLevel compressionLevel, final SpaceType spaceType) {
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevel(mode, compressionLevel);
        MethodComponentContext encoderContext = resolveEncoder(resolvedCompressionLevel);

        KNNEngine knnEngine = Mode.ON_DISK == mode || encoderContext != null ? KNNEngine.FAISS : KNNEngine.DEFAULT;

        if (encoderContext != null) {
            return new KNNMethodContext(
                knnEngine,
                spaceType,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(
                        METHOD_PARAMETER_M,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M,
                        METHOD_PARAMETER_EF_CONSTRUCTION,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                        METHOD_PARAMETER_EF_SEARCH,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                        METHOD_ENCODER_PARAMETER,
                        encoderContext
                    )
                )
            );
        }

        if (knnEngine == KNNEngine.FAISS) {
            return new KNNMethodContext(
                knnEngine,
                spaceType,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(
                        METHOD_PARAMETER_M,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M,
                        METHOD_PARAMETER_EF_CONSTRUCTION,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                        METHOD_PARAMETER_EF_SEARCH,
                        KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH
                    )
                )
            );
        }

        return new KNNMethodContext(
            knnEngine,
            spaceType,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(
                    METHOD_PARAMETER_M,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M,
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION
                )
            )
        );
    }

    private KNNMethodContext resolveWithTraining(Mode mode, CompressionLevel compressionLevel, SpaceType spaceType) {
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevel(mode, compressionLevel);
        MethodComponentContext encoderContext = resolveEncoder(resolvedCompressionLevel);
        if (encoderContext != null) {
            return new KNNMethodContext(
                KNNEngine.FAISS,
                spaceType,
                new MethodComponentContext(
                    METHOD_IVF,
                    Map.of(
                        METHOD_PARAMETER_NLIST,
                        METHOD_PARAMETER_NLIST_DEFAULT,
                        METHOD_PARAMETER_NPROBES,
                        METHOD_PARAMETER_NPROBES_DEFAULT,
                        METHOD_ENCODER_PARAMETER,
                        encoderContext
                    )
                )
            );
        }

        return new KNNMethodContext(
            KNNEngine.FAISS,
            spaceType,
            new MethodComponentContext(
                METHOD_IVF,
                Map.of(METHOD_PARAMETER_NLIST, METHOD_PARAMETER_NLIST_DEFAULT, METHOD_PARAMETER_NPROBES, METHOD_PARAMETER_NPROBES_DEFAULT)
            )
        );
    }

    /**
     * Resolves the rescore context give the {@link Mode} and {@link CompressionLevel}
     *
     * @param mode {@link Mode}
     * @param compressionLevel {@link CompressionLevel}
     * @return {@link RescoreContext}
     */
    public RescoreContext resolveRescoreContext(Mode mode, CompressionLevel compressionLevel) {
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevel(mode, compressionLevel);
        return resolvedCompressionLevel.getDefaultRescoreContext();
    }

    private CompressionLevel resolveCompressionLevel(Mode mode, CompressionLevel compressionLevel) {
        if (CompressionLevel.isConfigured(compressionLevel)) {
            return compressionLevel;
        }

        if (mode == Mode.ON_DISK) {
            return DEFAULT_COMPRESSION_FOR_MODE_ON_DISK;
        }

        return DEFAULT_COMPRESSION_FOR_MODE_IN_MEMORY;
    }

    private MethodComponentContext resolveEncoder(CompressionLevel compressionLevel) {
        if (CompressionLevel.isConfigured(compressionLevel) == false) {
            throw new IllegalStateException("Compression level needs to be configured");
        }

        if (SUPPORTED_COMPRESSION_LEVELS.contains(compressionLevel) == false) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Unsupported compression level: \"[%s]\"", compressionLevel.getName())
            );
        }

        if (compressionLevel == CompressionLevel.x1) {
            return null;
        }

        if (compressionLevel == CompressionLevel.x2) {
            return new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16, FAISS_SQ_CLIP, true));
        }

        return new MethodComponentContext(
            QFrameBitEncoder.NAME,
            Map.of(QFrameBitEncoder.BITCOUNT_PARAM, compressionLevel.numBitsForFloat32())
        );
    }

}
