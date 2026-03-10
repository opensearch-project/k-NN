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
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.engine.lucene.LuceneFlatMethod.FLAT_METHOD_COMPONENT;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.HNSW_METHOD_COMPONENT;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.SQ_ENCODER;

public class LuceneMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(CompressionLevel.x1, CompressionLevel.x4);
    private static final Set<CompressionLevel> FLAT_SUPPORTED_COMPRESSION_LEVELS = Set.of(CompressionLevel.x32);
    private static final CompressionLevel FLAT_DEFAULT_COMPRESSION = CompressionLevel.x32;

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        // Check if the user specified flat method
        if (isFlatMethod(knnMethodContext)) {
            validateNotTrainingContext(shouldRequireTraining, knnMethodConfigContext);
            return resolveFlatMethod(knnMethodContext, knnMethodConfigContext, spaceType);
        }

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

    private boolean isFlatMethod(KNNMethodContext knnMethodContext) {
        return knnMethodContext != null && METHOD_FLAT.equals(knnMethodContext.getMethodComponentContext().getName());
    }

    private ResolvedMethodContext resolveFlatMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        SpaceType spaceType
    ) {
        validateFlatMethodParameters(knnMethodContext);
        validateFlatMode(knnMethodConfigContext);
        validateFlatCompressionLevel(knnMethodConfigContext);

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.LUCENE,
            spaceType,
            METHOD_FLAT
        );
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, FLAT_METHOD_COMPONENT);

        CompressionLevel resolvedCompressionLevel = CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())
            ? knnMethodConfigContext.getCompressionLevel()
            : FLAT_DEFAULT_COMPRESSION;

        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    private void validateFlatMethodParameters(KNNMethodContext knnMethodContext) {
        Map<String, Object> parameters = knnMethodContext.getMethodComponentContext().getParameters();
        if (parameters != null && !parameters.isEmpty()) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(Locale.ROOT, "Parameters are not supported for the \"%s\" method", METHOD_FLAT)
            );
            throw validationException;
        }
    }

    private void validateFlatMode(KNNMethodConfigContext knnMethodConfigContext) {
        if (Mode.isConfigured(knnMethodConfigContext.getMode())) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(Locale.ROOT, "\"%s\" is not supported for the \"%s\" method", MODE_PARAMETER, METHOD_FLAT)
            );
            throw validationException;
        }
    }

    private void validateFlatCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        if (CompressionLevel.isConfigured(compressionLevel) && !FLAT_SUPPORTED_COMPRESSION_LEVELS.contains(compressionLevel)) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "\"%s\" method only supports \"%s\" compression",
                    METHOD_FLAT,
                    FLAT_DEFAULT_COMPRESSION.getName()
                )
            );
            throw validationException;
        }
    }

    private void validateNotTrainingContext(boolean shouldRequireTraining, KNNMethodConfigContext knnMethodConfigContext) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, KNNEngine.LUCENE, null);
        if (validationException != null) {
            throw validationException;
        }
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
        MethodComponentContext encoderComponentContext = new MethodComponentContext(SQ_ENCODER.getName(), new HashMap<>());
        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            SQ_ENCODER.getMethodComponent(),
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
