/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.index.engine.lucene.LuceneFlatMethod.FLAT_METHOD_COMPONENT;

/**
 * Resolves method configuration for the Lucene flat method. The flat method uses SQ (1-bit quantization)
 * without an HNSW graph, supporting only {@link org.opensearch.knn.index.mapper.CompressionLevel#x32} compression
 * and does not support {@link org.opensearch.knn.index.mapper.Mode}.
 */
public class LuceneFlatMethodResolver extends AbstractMethodResolver {

    static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(CompressionLevel.x32);
    static final CompressionLevel DEFAULT_COMPRESSION = CompressionLevel.x32;

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        validateNotTrainingContext(shouldRequireTraining, knnMethodConfigContext);
        validateParameters(knnMethodContext);
        validateMode(knnMethodConfigContext);

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            BuiltinKNNEngine.LUCENE,
            spaceType,
            METHOD_FLAT
        );
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, FLAT_METHOD_COMPONENT);

        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(validateAndResolveCompressionLevel(knnMethodConfigContext))
            .build();
    }

    private void validateNotTrainingContext(boolean shouldRequireTraining, KNNMethodConfigContext knnMethodConfigContext) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, BuiltinKNNEngine.LUCENE, null);
        if (validationException != null) {
            throw validationException;
        }
    }

    private void validateParameters(KNNMethodContext knnMethodContext) {
        Map<String, Object> parameters = knnMethodContext.getMethodComponentContext().getParameters();
        if (parameters != null && !parameters.isEmpty()) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(Locale.ROOT, "Parameters are not supported for the \"%s\" method", METHOD_FLAT)
            );
            throw validationException;
        }
    }

    private void validateMode(KNNMethodConfigContext knnMethodConfigContext) {
        if (Mode.isConfigured(knnMethodConfigContext.getMode())) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(Locale.ROOT, "\"%s\" is not supported for the \"%s\" method", MODE_PARAMETER, METHOD_FLAT)
            );
            throw validationException;
        }
    }

    private CompressionLevel validateAndResolveCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        if (CompressionLevel.isConfigured(compressionLevel)) {
            if (!SUPPORTED_COMPRESSION_LEVELS.contains(compressionLevel)) {
                ValidationException validationException = new ValidationException();
                validationException.addValidationError(
                    String.format(Locale.ROOT, "\"%s\" method only supports \"%s\" compression", METHOD_FLAT, DEFAULT_COMPRESSION.getName())
                );
                throw validationException;
            }
            return compressionLevel;
        }
        return DEFAULT_COMPRESSION;
    }
}
