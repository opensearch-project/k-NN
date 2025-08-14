/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.HNSW_METHOD_COMPONENT;

public class LuceneMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x4,
        CompressionLevel.x32
    );

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        validateConfig(knnMethodConfigContext, shouldRequireTraining);
        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.LUCENE,
            spaceType,
            METHOD_HNSW
        );
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext);
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, HNSW_METHOD_COMPONENT);
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevelFromMethodContext(
            resolvedKNNMethodContext,
            knnMethodConfigContext,
            LuceneHNSWMethod.SUPPORTED_ENCODERS
        );
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompressionLevel);
        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    protected void resolveEncoder(KNNMethodContext resolvedKNNMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (shouldEncoderBeResolved(resolvedKNNMethodContext, knnMethodConfigContext) == false) {
            return;
        }

        CompressionLevel resolvedCompressionLevel = getDefaultCompressionLevel(knnMethodConfigContext);
        if (resolvedCompressionLevel == CompressionLevel.x1) {
            return;
        }

        MethodComponentContext methodComponentContext = resolvedKNNMethodContext.getMethodComponentContext();

        String encoderName = (resolvedCompressionLevel == CompressionLevel.x32)
            ? LuceneHNSWMethod.BBQ_ENCODER.getName()
            : LuceneHNSWMethod.SQ_ENCODER.getName();
        MethodComponent encoderComponent = (resolvedCompressionLevel == CompressionLevel.x32)
            ? LuceneHNSWMethod.BBQ_ENCODER.getMethodComponent()
            : LuceneHNSWMethod.SQ_ENCODER.getMethodComponent();

        MethodComponentContext encoderComponentContext = new MethodComponentContext(encoderName, new HashMap<>());
        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoderComponent,
            knnMethodConfigContext
        );
        encoderComponentContext.getParameters().putAll(resolvedParams);
        methodComponentContext.getParameters().put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
    }

    // Method validates for explicit contradictions in the config
    private void validateConfig(KNNMethodConfigContext knnMethodConfigContext, boolean shouldRequireTraining) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, KNNEngine.LUCENE, null);
        validationException = validateCompressionSupported(
            knnMethodConfigContext.getCompressionLevel(),
            SUPPORTED_COMPRESSION_LEVELS,
            KNNEngine.LUCENE,
            validationException
        );
        validationException = validateCompressionNotx1WhenOnDisk(knnMethodConfigContext, validationException);
        if (validationException != null) {
            throw validationException;
        }
    }

    private CompressionLevel getDefaultCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }
        if (knnMethodConfigContext.getMode() == Mode.ON_DISK) {
            return CompressionLevel.x4;
        }
        return CompressionLevel.x1;
    }
}
