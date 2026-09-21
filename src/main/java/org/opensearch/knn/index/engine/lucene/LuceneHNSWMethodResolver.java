/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
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

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
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
 *
 * <p>Those levels are measured against FLOAT's 32-bit storage. {@code half_float} supports only x1
 * and x16, and its x16 is SQ <b>1-bit</b> — 16 bits down to 1 — not the x32 level bits=1 denotes for
 * FLOAT.
 */
public class LuceneHNSWMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x4,
        CompressionLevel.x32
    );
    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT = Set.of(CompressionLevel.x1, CompressionLevel.x16);
    static final CompressionLevel DEFAULT_COMPRESSION_HALF_FLOAT = CompressionLevel.x1;

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        if (VectorDataType.HALF_FLOAT == knnMethodConfigContext.getVectorDataType()) {
            return resolveHalfFloatMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
        }

        validateConfig(knnMethodConfigContext, shouldRequireTraining);
        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.LUCENE,
            spaceType,
            METHOD_HNSW
        );
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext);
        resolveEncoderBitsAndValidate(knnMethodContext, resolvedKNNMethodContext, knnMethodConfigContext);
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

    private ResolvedMethodContext resolveHalfFloatMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        ValidationException validationException = validateNotTrainingContext(shouldRequireTraining, KNNEngine.LUCENE, null);
        validationException = validateCompressionSupported(
            knnMethodConfigContext.getCompressionLevel(),
            SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT,
            KNNEngine.LUCENE,
            knnMethodConfigContext.getVectorDataType(),
            validationException
        );
        // half_float's only encoder configuration (SQ 1-bit) is fully determined by compression_level
        // (x16) and auto-resolved internally - there's no tunable parameter surface to expose, so
        // users configure it via compression_level, not by writing an encoder block themselves.
        if (isEncoderSpecified(knnMethodContext)) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "\"%s\" parameter is not supported for \"%s\" data type; use \"%s\" instead.",
                    METHOD_ENCODER_PARAMETER,
                    VectorDataType.HALF_FLOAT.getValue(),
                    COMPRESSION_LEVEL_PARAMETER
                )
            );
        }
        validationException = validateCompressionNotx1WhenOnDisk(knnMethodConfigContext, validationException);
        if (validationException != null) {
            throw validationException;
        }

        KNNMethodContext resolvedKNNMethodContext = initResolvedKNNMethodContext(
            knnMethodContext,
            KNNEngine.LUCENE,
            spaceType,
            METHOD_HNSW
        );
        resolveEncoder(resolvedKNNMethodContext, knnMethodConfigContext);
        resolveEncoderBitsAndValidate(knnMethodContext, resolvedKNNMethodContext, knnMethodConfigContext);
        resolveMethodParams(resolvedKNNMethodContext.getMethodComponentContext(), knnMethodConfigContext, HNSW_METHOD_COMPONENT);

        CompressionLevel resolvedCompressionLevel = isEncoderSpecified(resolvedKNNMethodContext)
            ? resolveCompressionLevelFromMethodContext(
                resolvedKNNMethodContext,
                knnMethodConfigContext,
                LuceneHNSWMethod.SUPPORTED_ENCODERS
            )
            : getDataTypeAwareDefaultCompressionLevel(knnMethodConfigContext);
        validateCompressionConflicts(knnMethodConfigContext.getCompressionLevel(), resolvedCompressionLevel);
        return ResolvedMethodContext.builder()
            .knnMethodContext(resolvedKNNMethodContext)
            .compressionLevel(resolvedCompressionLevel)
            .build();
    }

    @Override
    protected boolean shouldEncoderBeResolved(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (isEncoderSpecified(knnMethodContext)) {
            return false;
        }

        if (knnMethodConfigContext.getVectorDataType() == VectorDataType.HALF_FLOAT) {
            return getDataTypeAwareDefaultCompressionLevel(knnMethodConfigContext) == CompressionLevel.x16;
        }

        return super.shouldEncoderBeResolved(knnMethodContext, knnMethodConfigContext);
    }

    protected void resolveEncoder(KNNMethodContext resolvedKNNMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (shouldEncoderBeResolved(resolvedKNNMethodContext, knnMethodConfigContext) == false) {
            return;
        }

        CompressionLevel resolvedCompressionLevel = getDataTypeAwareDefaultCompressionLevel(knnMethodConfigContext);
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
    private void resolveEncoderBitsAndValidate(
        KNNMethodContext originalMethodContext,
        KNNMethodContext resolvedKNNMethodContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        if (!isEncoderSpecified(resolvedKNNMethodContext)) {
            return;
        }
        boolean didUserSpecifyEncoder = isEncoderSpecified(originalMethodContext);
        boolean isV360OrLater = knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0);

        MethodComponentContext encoderComponentContext = getEncoderComponentContext(resolvedKNNMethodContext);
        if (encoderComponentContext == null) {
            return;
        }

        boolean bitsAlreadySet = encoderComponentContext.getParameters().containsKey(LUCENE_SQ_BITS);
        boolean skipAutoResolve = isV360OrLater && didUserSpecifyEncoder;

        if (bitsAlreadySet == false && skipAutoResolve == false) {
            CompressionLevel effectiveCompression = CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())
                ? knnMethodConfigContext.getCompressionLevel()
                : getDataTypeAwareDefaultCompressionLevel(knnMethodConfigContext);
            // LUCENE_SCALAR_QUANTIZER_DEFAULT_BITS_AFTER_V360 is 1, and half_float's SQ 1-bit level is x16 rather
            // than FLOAT's x32, so the comparison has to be made against the data type's own scale.
            boolean useNewDefault = isV360OrLater
                && LuceneSQEncoder.Bits.fromValue(LUCENE_SCALAR_QUANTIZER_DEFAULT_BITS_AFTER_V360)
                    .getCompressionLevel(knnMethodConfigContext.getVectorDataType()) == effectiveCompression;
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
            knnMethodConfigContext.getVectorDataType(),
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

    /**
     * Defers to {@link #getDefaultCompressionLevel} for every data type but {@code half_float}, whose
     * ON_DISK default is x16 (its SQ 1-bit level) rather than FLOAT's x32.
     */
    private CompressionLevel getDataTypeAwareDefaultCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext.getVectorDataType() != VectorDataType.HALF_FLOAT) {
            return getDefaultCompressionLevel(knnMethodConfigContext);
        }
        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }
        return Mode.ON_DISK == knnMethodConfigContext.getMode() ? CompressionLevel.x16 : DEFAULT_COMPRESSION_HALF_FLOAT;
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

            // half_float only supports the 1-bit path; 7 stays float-only.
            if (configContext.getVectorDataType() == VectorDataType.HALF_FLOAT && bits != Bits.ONE.getValue()) {
                validationException.addValidationError(
                    String.format(
                        Locale.ROOT,
                        "[%s] data type only supports [%s=%d] for encoder [%s].",
                        VectorDataType.HALF_FLOAT.getValue(),
                        LUCENE_SQ_BITS,
                        Bits.ONE.getValue(),
                        ENCODER_SQ
                    )
                );
                throw validationException;
            }

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
                CompressionLevel expectedCompression = Bits.fromValue(bits).getCompressionLevel(configContext.getVectorDataType());
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
