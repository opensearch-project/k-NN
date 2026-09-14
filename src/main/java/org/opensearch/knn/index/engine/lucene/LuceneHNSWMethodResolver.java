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
import org.opensearch.knn.index.engine.Encoder.QuantizationBits;
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

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.HNSW_METHOD_COMPONENT;
import static org.opensearch.knn.index.engine.lucene.LuceneHNSWMethod.SUPPORTED_ENCODERS;

/**
 * Resolves method configuration for the Lucene HNSW method. Supports optional scalar quantization
 * encoding and {@link org.opensearch.knn.index.mapper.Mode}-based compression resolution. Supported
 * compression levels are {@link org.opensearch.knn.index.mapper.CompressionLevel#x1} (raw),
 * {@link org.opensearch.knn.index.mapper.CompressionLevel#x4} (SQ 7-bit, legacy),
 * {@link org.opensearch.knn.index.mapper.CompressionLevel#x8} (SQ 4-bit),
 * {@link org.opensearch.knn.index.mapper.CompressionLevel#x16} (SQ 2-bit), and
 * {@link org.opensearch.knn.index.mapper.CompressionLevel#x32} (SQ 1-bit). The 1-bit path
 * requires indices created on or after 3.6.0; the 2/4-bit paths require indices created on or
 * after {@link org.opensearch.knn.common.KNNConstants#LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION}.
 *
 * <p>Those levels are measured against FLOAT's 32-bit storage. {@code half_float} supports only x1
 * and x16, and its x16 is SQ <b>1-bit</b> — 16 bits down to 1 — not the 2-bit level x16 denotes for
 * FLOAT.
 */
public class LuceneHNSWMethodResolver extends AbstractMethodResolver {

    private static final Set<CompressionLevel> SUPPORTED_COMPRESSION_LEVELS = Set.of(
        CompressionLevel.x1,
        CompressionLevel.x4,
        CompressionLevel.x8,
        CompressionLevel.x16,
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
            // pre-3.6.0 → only the legacy 7-bit path (x4). 3.6.0+ → derive from compression.
            int resolvedBits = isV360OrLater
                ? QuantizationBits.fromCompressionLevel(effectiveCompression, knnMethodConfigContext.getVectorDataType()).getValue()
                : LUCENE_SQ_DEFAULT_BITS;
            encoderComponentContext.getParameters().put(LUCENE_SQ_BITS, resolvedBits);
        }
        String encoderName = encoderComponentContext.getName();
        Encoder encoder = SUPPORTED_ENCODERS.get(encoderName);

        if (encoder == null) {
            return;
        }
        encoder.validate(resolvedKNNMethodContext, knnMethodConfigContext);
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
        validationException = validateMultiBitCompressionVersion(knnMethodConfigContext, validationException);
        validationException = validateCompressionNotx1WhenOnDisk(knnMethodConfigContext, validationException);
        if (validationException != null) {
            throw validationException;
        }
    }

    /**
     * Rejects x8 / x16 compression on the Lucene HNSW method for indices created before
     * {@link org.opensearch.knn.common.KNNConstants#LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION}. The
     * 2-bit and 4-bit scalar-quantization codec files did not exist in earlier codecs, so an
     * older index cannot read them; reject the mapping up front rather than deferring to a
     * codec-time failure.
     */
    private ValidationException validateMultiBitCompressionVersion(
        KNNMethodConfigContext knnMethodConfigContext,
        ValidationException validationException
    ) {
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        if (compressionLevel != CompressionLevel.x8 && compressionLevel != CompressionLevel.x16) {
            return validationException;
        }
        Version versionCreated = knnMethodConfigContext.getVersionCreated();
        if (versionCreated == null || versionCreated.onOrAfter(LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION)) {
            return validationException;
        }
        validationException = validationException == null ? new ValidationException() : validationException;
        validationException.addValidationError(
            String.format(
                Locale.ROOT,
                "\"%s\" compression on the [%s] method for engine [%s] requires an index created with version %s or later",
                compressionLevel.getName(),
                METHOD_HNSW,
                KNNEngine.LUCENE.getName(),
                LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION
            )
        );
        return validationException;
    }

    private CompressionLevel getDefaultCompressionLevel(KNNMethodConfigContext knnMethodConfigContext) {
        return getDefaultCompressionLevel(knnMethodConfigContext, CompressionLevel.x4);
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
}
