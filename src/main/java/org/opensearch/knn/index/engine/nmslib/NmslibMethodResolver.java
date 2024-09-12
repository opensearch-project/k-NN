/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.nmslib;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.engine.nmslib.NmslibHNSWMethod.HNSW_METHOD_COMPONENT;

/**
 * Method resolution logic for nmslib. Because nmslib does not support quantization, it is in general a validation
 * before returning the original request
 */
public class NmslibMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(CompressionLevel.x1);

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
            KNNEngine.NMSLIB,
            spaceType,
            METHOD_HNSW
        );
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, HNSW_METHOD_COMPONENT);
        return ResolvedMethodContext.builder().knnMethodContext(resolvedKNNMethodContext).compressionLevel(CompressionLevel.x1).build();
    }

    // Method validates for explicit contradictions in the config
    private void validateConfig(KNNMethodConfigContext knnMethodConfigContext, boolean shouldRequireTraining) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, KNNEngine.NMSLIB, null);
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        validationException = validateCompressionSupported(
            compressionLevel,
            SUPPORTED_COMPRESSION_LEVELS,
            KNNEngine.NMSLIB,
            validationException
        );

        if (Mode.ON_DISK == knnMethodConfigContext.getMode()) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError("Nmslib engine does not support disk-based search");
        }

        if (validationException != null) {
            throw validationException;
        }
    }
}
