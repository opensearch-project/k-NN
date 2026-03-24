/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.Version;
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
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SCALAR_QUANTIZER_DEFAULT_BITS_AFTER_V360;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.HNSW_METHOD_COMPONENT;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.SUPPORTED_ENCODERS;
import static org.opensearch.knn.index.engine.lucene.LuceneSQEncoder.Bits;
import static org.opensearch.knn.index.engine.lucene.LuceneSQEncoder.LUCENE_PRE_360_SUPPORTED_SQ_BITS;
import static org.opensearch.knn.index.engine.lucene.LuceneSQEncoder.LUCENE_SQ_BITS_SUPPORTED;

/**
 * Resolves method configuration for the Lucene HNSW method. Supports optional scalar quantization
 * encoding and {@link org.opensearch.knn.index.mapper.Mode}-based compression resolution, with
 * supported compression levels of {@link org.opensearch.knn.index.mapper.CompressionLevel#x1} and
 * {@link org.opensearch.knn.index.mapper.CompressionLevel#x4}.
 */
public class LuceneHNSWMethodResolver extends AbstractMethodResolver {

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
        resolveEncoderBitsAndValidate(resolvedKNNMethodContext, knnMethodConfigContext);
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

        String encoderName;
        MethodComponent encoderComponent;

        encoderName = LuceneHNSWMethod.SQ_ENCODER.getName();
        encoderComponent = LuceneHNSWMethod.SQ_ENCODER.getMethodComponent();

        MethodComponentContext encoderComponentContext = new MethodComponentContext(encoderName, new HashMap<>());
        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoderComponent,
            knnMethodConfigContext
        );

        encoderComponentContext.getParameters().putAll(resolvedParams);
        methodComponentContext.getParameters().put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
    }

    // if encoder gets resolved, determine if default bits need to be added and validate encoder config makes sense
    private void resolveEncoderBitsAndValidate(KNNMethodContext resolvedKNNMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (!isEncoderSpecified(resolvedKNNMethodContext)) {
            return;
        }
        MethodComponentContext encoderComponentContext = getEncoderComponentContext(resolvedKNNMethodContext);
        if (encoderComponentContext == null) {
            return;
        }
        if (!encoderComponentContext.getParameters().containsKey(LUCENE_SQ_BITS)) {
            CompressionLevel effectiveCompression = CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())
                ? knnMethodConfigContext.getCompressionLevel()
                : getDefaultCompressionLevel(knnMethodConfigContext);
            boolean useNewDefault = knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)
                && LuceneSQEncoder.Bits.fromValue(LUCENE_SCALAR_QUANTIZER_DEFAULT_BITS_AFTER_V360)
                    .getCompressionLevel() == effectiveCompression;
            encoderComponentContext.getParameters()
                .put(LUCENE_SQ_BITS, useNewDefault ? LUCENE_SCALAR_QUANTIZER_DEFAULT_BITS_AFTER_V360 : LUCENE_SQ_DEFAULT_BITS);
        }
        String encoderName = encoderComponentContext.getName();
        Encoder encoder = SUPPORTED_ENCODERS.get(encoderName);

        // Skip the additional validation at the end if not using SQ
        // TODO: Once validateEncoderParams is defined as an interface method, we can clean this up
        if (encoder == null || !encoderName.equals(ENCODER_SQ)) {
            return;
        }
        validateEncoderParams(resolvedKNNMethodContext, knnMethodConfigContext);
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
            // Starting with version 3.6, supporting 32x compression by default
            if (Version.V_3_6_0.onOrBefore(knnMethodConfigContext.getVersionCreated())) {
                return CompressionLevel.x32;
            }
            return CompressionLevel.x4;
        }
        return CompressionLevel.x1;
    }

    // TODO: The Encoder interface currently only has validateEncoderConfig() which uses
    // TrainingConfigValidation* types designed for model training. We should add a general-purpose
    // validation method to the Encoder interface (e.g. Encoder.validate(KNNMethodContext, KNNMethodConfigContext))
    // that both Faiss and Lucene resolvers can delegate to, decoupled from training concerns.
    // See: FaissSQEncoder.validateEncoderConfig() for the Faiss equivalent of this validation.
    protected static void validateEncoderParams(KNNMethodContext resolvedMethodContext, KNNMethodConfigContext configContext) {
        if (resolvedMethodContext == null || configContext == null) {
            return;
        }

        MethodComponentContext encoderContext = (MethodComponentContext) resolvedMethodContext.getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        if (encoderContext == null) {
            return;
        }

        Map<String, Object> encoderParams = encoderContext.getParameters();
        Version version = configContext.getVersionCreated();
        boolean isV360OrLater = version != null && version.onOrAfter(Version.V_3_6_0);
        Object bitsObj = encoderParams.get(LUCENE_SQ_BITS);
        Set<String> nonBitParameters = encoderParams.keySet().stream().filter(k -> !k.equals(LUCENE_SQ_BITS)).collect(Collectors.toSet());

        ValidationException validationException = new ValidationException();

        // On 3.6.0+, bits is required when the user explicitly specifies the lucene sq encoder
        if (isV360OrLater && bitsObj == null) {
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Parameter [%s] is required for encoder [%s] on indices created with version 3.6.0 or later. " + "Supported values: %s",
                    LUCENE_SQ_BITS,
                    ENCODER_SQ,
                    LUCENE_SQ_BITS_SUPPORTED
                )
            );
            throw validationException;
        }

        if (bitsObj instanceof Integer) {
            int bits = (Integer) bitsObj;

            // bits=1 does not support other parameters
            if (bits == Bits.ONE.getValue()) {
                if (!nonBitParameters.isEmpty()) {
                    validationException.addValidationError(
                        String.format(
                            Locale.ROOT,
                            "Parameters [%s] are not supported when [%s=%d] for encoder [%s]. "
                                + "The 1-bit scalar quantization path does not use additional parameters.",
                            nonBitParameters,
                            LUCENE_SQ_BITS,
                            bits,
                            ENCODER_SQ
                        )
                    );
                    throw validationException;
                }
                if (!isV360OrLater) {
                    validationException.addValidationError(
                        String.format(
                            Locale.ROOT,
                            "Parameter [%s=%d] is only supported for indices created with version 3.6.0 or later. "
                                + "Supported values: %s",
                            LUCENE_SQ_BITS,
                            bits,
                            LUCENE_PRE_360_SUPPORTED_SQ_BITS
                        )
                    );
                    throw validationException;
                }
            }

            // Validate compression level compatibility if explicitly set
            CompressionLevel configuredCompression = configContext.getCompressionLevel();
            if (CompressionLevel.isConfigured(configuredCompression)) {
                CompressionLevel expectedCompression = Bits.fromValue(bits).getCompressionLevel();
                if (configuredCompression != expectedCompression) {
                    validationException.addValidationError(
                        String.format(
                            Locale.ROOT,
                            "Compression level [%s] is incompatible with [%s=%d] for encoder [%s]. " + "Expected compression level: [%s]",
                            configuredCompression.getName(),
                            LUCENE_SQ_BITS,
                            bits,
                            ENCODER_SQ,
                            expectedCompression.getName()
                        )
                    );
                    throw validationException;
                }
            }
        }
    }
}
