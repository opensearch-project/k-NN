/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.index.engine.faiss.FaissHNSWMethod.HNSW_COMPONENT;
import static org.opensearch.knn.index.engine.faiss.FaissIVFMethod.IVF_COMPONENT;

public class FaissMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x2,
        CompressionLevel.x8,
        CompressionLevel.x16,
        CompressionLevel.x32
    );

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        // Initial validation to ensure that there are no contradictions in provided parameters
        validateConfig(knnMethodConfigContext);

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.FAISS,
            spaceType,
            shouldRequireTraining ? METHOD_IVF : METHOD_HNSW
        );
        MethodComponent method = METHOD_HNSW.equals(resolvedKNNMethodContext.getMethodComponentContext().getName()) == false
            ? IVF_COMPONENT
            : HNSW_COMPONENT;
        Map<String, Encoder> encoderMap = method == HNSW_COMPONENT ? FaissHNSWMethod.SUPPORTED_ENCODERS : FaissIVFMethod.SUPPORTED_ENCODERS;

        // Fill in parameters for the encoder and then the method.
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap);
        // From the resolved method context, get the compression level and validate it against the passed in
        // configuration
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevelFromMethodContext(
            resolvedKNNMethodContext,
            knnMethodConfigContext,
            encoderMap
        );

        // Validate encoder parameters
        validateEncoderConfig(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap);

        // Validate that resolved compression doesnt have any conflicts
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompressionLevel);
        knnMethodConfigContext.setCompressionLevel(resolvedCompressionLevel);
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, method);

        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    private void resolveEncoder(
        KNNMethodContext resolvedKNNMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap
    ) {
        if (shouldEncoderBeResolved(resolvedKNNMethodContext, knnMethodConfigContext) == false) {
            return;
        }

        CompressionLevel resolvedCompressionLevel = getDefaultCompressionLevel(knnMethodConfigContext);
        if (resolvedCompressionLevel == CompressionLevel.x1) {
            return;
        }

        MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_FLAT, new HashMap<>());
        Encoder encoder = encoderMap.get(ENCODER_FLAT);
        if (CompressionLevel.x2 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
            encoder = encoderMap.get(ENCODER_SQ);
            encoderComponentContext.getParameters().put(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16);
        }

        if (CompressionLevel.x8 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
            encoder = encoderMap.get(QFrameBitEncoder.NAME);
            encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x8.numBitsForFloat32());
        }

        if (CompressionLevel.x16 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
            encoder = encoderMap.get(QFrameBitEncoder.NAME);
            encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x16.numBitsForFloat32());
        }

        if (CompressionLevel.x32 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
            encoder = encoderMap.get(QFrameBitEncoder.NAME);
            encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x32.numBitsForFloat32());
        }

        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoder.getMethodComponent(),
            knnMethodConfigContext
        );
        encoderComponentContext.getParameters().putAll(resolvedParams);
        resolvedKNNMethodContext.getMethodComponentContext().getParameters().put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
    }

    // Method validates for explicit contradictions in the config
    private void validateConfig(KNNMethodConfigContext knnMethodConfigContext) {
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        ValidationException validationException = validateCompressionSupported(
            compressionLevel,
            SUPPORTED_COMPRESSION_LEVELS,
            KNNEngine.FAISS,
            null
        );
        validationException = validateCompressionNotx1WhenOnDisk(knnMethodConfigContext, validationException);
        if (validationException != null) {
            throw validationException;
        }
    }

    protected void validateEncoderConfig(
        KNNMethodContext resolvedKnnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap
    ) {
        if (isEncoderSpecified(resolvedKnnMethodContext) == false) {
            return;
        }
        Encoder encoder = encoderMap.get(getEncoderName(resolvedKnnMethodContext));
        if (encoder == null) {
            return;
        }

        TrainingConfigValidationInput.TrainingConfigValidationInputBuilder inputBuilder = TrainingConfigValidationInput.builder();

        TrainingConfigValidationOutput validationOutput = encoder.validateEncoderConfig(
            inputBuilder.knnMethodContext(resolvedKnnMethodContext).knnMethodConfigContext(knnMethodConfigContext).build()
        );

        if (validationOutput.getValid() != null && !validationOutput.getValid()) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(validationOutput.getErrorMessage());
            throw validationException;
        }
    }

    private CompressionLevel getDefaultCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }
        if (knnMethodConfigContext.getMode() == Mode.ON_DISK) {
            return CompressionLevel.x32;
        }
        return CompressionLevel.x1;
    }
}
