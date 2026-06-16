/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

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
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE_FP16;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_PRIMARY_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_RESIDUAL_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_SVS_VAMANA;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;

/**
 * {@link org.opensearch.knn.index.engine.MethodResolver} for the experimental SVS engine. The sole method is
 * {@code svs_vamana}; this resolver fills in its defaults, maps a user-supplied {@code compression_level} to
 * the SVS encoders ({@code 2x} -> {@code sq}(fp16), {@code 4x} -> {@code lvq}(4,4), {@code 8x} ->
 * {@code lvq}(4,0)) and rejects configurations SVS cannot serve ({@code on_disk} mode, training contexts,
 * unsupported compression levels). It lives entirely in the sandbox: the core Faiss resolver knows nothing
 * about SVS.
 */
@ExperimentalAlgorithm(description = "Intel SVS Vamana method resolver", since = "3.7.0")
public class SvsMethodResolver extends AbstractMethodResolver {

    // x4 maps to lvq(4,4); the core engines top out differently (no x4) and support higher levels SVS does not.
    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x2,
        CompressionLevel.x4,
        CompressionLevel.x8
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
            KNNEngine.EXPERIMENTAL,
            spaceType,
            METHOD_SVS_VAMANA
        );

        String methodName = resolvedKNNMethodContext.getMethodComponentContext().getName();
        if (METHOD_SVS_VAMANA.equals(methodName) == false) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Invalid method name [%s] for engine [%s]. The only supported method is [%s].",
                    methodName,
                    KNNEngine.EXPERIMENTAL.getName(),
                    METHOD_SVS_VAMANA
                )
            );
            throw validationException;
        }

        Map<String, Encoder> encoderMap = FaissSVSVamanaMethod.SUPPORTED_ENCODERS;

        // Fill in parameters for the encoder and then the method.
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap);
        CompressionLevel resolvedCompressionLevel = resolveCompressionLevelFromMethodContext(
            resolvedKNNMethodContext,
            knnMethodConfigContext,
            encoderMap
        );

        validateEncoderConfig(resolvedKNNMethodContext, knnMethodConfigContext, encoderMap);
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompressionLevel);
        knnMethodConfigContext.setCompressionLevel(resolvedCompressionLevel);
        resolveMethodParams(
            resolvedKNNMethodContext.getMethodComponentContext(),
            knnMethodConfigContext,
            FaissSVSVamanaMethod.METHOD_COMPONENT
        );

        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    private void validateConfig(KNNMethodConfigContext knnMethodConfigContext, boolean shouldRequireTraining) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, KNNEngine.EXPERIMENTAL, null);
        validationException = validateCompressionSupported(
            knnMethodConfigContext.getCompressionLevel(),
            SUPPORTED_COMPRESSION_LEVELS,
            KNNEngine.EXPERIMENTAL,
            validationException
        );
        if (knnMethodConfigContext.getMode() == Mode.ON_DISK) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "mode=on_disk is not supported with %s; SVS is an in-memory execution path. "
                        + "Use mode=in_memory or a different method.",
                    METHOD_SVS_VAMANA
                )
            );
        }
        if (validationException != null) {
            throw validationException;
        }
    }

    /**
     * Resolves the SVS encoder for a user-supplied {@code compression_level} when no explicit encoder was
     * given. Other levels require an explicit {@code encoder} block.
     */
    private void resolveEncoder(
        KNNMethodContext resolvedKNNMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap
    ) {
        if (shouldEncoderBeResolved(resolvedKNNMethodContext, knnMethodConfigContext) == false) {
            return;
        }

        CompressionLevel resolvedCompressionLevel = CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())
            ? knnMethodConfigContext.getCompressionLevel()
            : CompressionLevel.x1;
        if (resolvedCompressionLevel == CompressionLevel.x1) {
            return;
        }

        MethodComponentContext encoderComponentContext;
        Encoder encoder;
        if (CompressionLevel.x2 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
            encoderComponentContext.getParameters().put(FAISS_SVS_SQ_TYPE, FAISS_SVS_SQ_TYPE_FP16);
            encoder = encoderMap.get(ENCODER_SQ);
        } else if (CompressionLevel.x4 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(FAISS_SVS_ENCODER_LVQ, new HashMap<>());
            encoderComponentContext.getParameters().put(METHOD_PARAMETER_LVQ_PRIMARY_BITS, 4);
            encoderComponentContext.getParameters().put(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, 4);
            encoder = encoderMap.get(FAISS_SVS_ENCODER_LVQ);
        } else if (CompressionLevel.x8 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(FAISS_SVS_ENCODER_LVQ, new HashMap<>());
            encoderComponentContext.getParameters().put(METHOD_PARAMETER_LVQ_PRIMARY_BITS, 4);
            encoderComponentContext.getParameters().put(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, 0);
            encoder = encoderMap.get(FAISS_SVS_ENCODER_LVQ);
        } else {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Compression level [%s] is not supported for %s via compression_level. "
                        + "Supported levels are 2x, 4x, 8x; for other levels specify an explicit encoder.",
                    resolvedCompressionLevel.getName(),
                    METHOD_SVS_VAMANA
                )
            );
            throw validationException;
        }

        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoder.getMethodComponent(),
            knnMethodConfigContext
        );
        encoderComponentContext.getParameters().putAll(resolvedParams);
        resolvedKNNMethodContext.getMethodComponentContext().getParameters().put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
    }

    private void validateEncoderConfig(
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
}
