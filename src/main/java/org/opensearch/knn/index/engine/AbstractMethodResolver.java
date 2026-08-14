/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.KNN_DEFAULT_COMPRESSION_FLIP_VERSION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Abstract {@link MethodResolver} with helpful utilitiy functions that can be shared across different
 * implementations
 */
public abstract class AbstractMethodResolver implements MethodResolver {

    /**
     * Utility method to get the compression level from the context
     *
     * @param resolvedKnnMethodContext Resolved method context. Should have an encoder set in the params if available
     * @return {@link CompressionLevel} Compression level that is configured with the {@link KNNMethodContext}
     */
    protected CompressionLevel resolveCompressionLevelFromMethodContext(
        KNNMethodContext resolvedKnnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        Map<String, Encoder> encoderMap
    ) {
        // If the context is null, the compression is not configured or the encoder is not defined, return not configured
        // because the method context does not contain this info
        if (isEncoderSpecified(resolvedKnnMethodContext) == false) {
            return CompressionLevel.x1;
        }
        Encoder encoder = encoderMap.get(getEncoderName(resolvedKnnMethodContext));
        if (encoder == null) {
            return CompressionLevel.NOT_CONFIGURED;
        }
        return encoder.calculateCompressionLevel(getEncoderComponentContext(resolvedKnnMethodContext), knnMethodConfigContext);
    }

    protected void resolveMethodParams(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext,
        MethodComponent methodComponent
    ) {
        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            methodComponentContext,
            methodComponent,
            knnMethodConfigContext
        );
        methodComponentContext.getParameters().putAll(resolvedParams);
    }

    protected KNNMethodContext initResolvedKNNMethodContext(
        KNNMethodContext originalMethodContext,
        KNNEngine knnEngine,
        SpaceType spaceType,
        String methodName
    ) {
        if (originalMethodContext == null) {
            return new KNNMethodContext(knnEngine, spaceType, new MethodComponentContext(methodName, new HashMap<>()));
        }
        return new KNNMethodContext(originalMethodContext);
    }

    protected String getEncoderName(KNNMethodContext knnMethodContext) {
        if (isEncoderSpecified(knnMethodContext) == false) {
            return null;
        }

        MethodComponentContext methodComponentContext = getEncoderComponentContext(knnMethodContext);
        if (methodComponentContext == null) {
            return null;
        }

        return methodComponentContext.getName();
    }

    protected MethodComponentContext getEncoderComponentContext(KNNMethodContext knnMethodContext) {
        if (isEncoderSpecified(knnMethodContext) == false) {
            return null;
        }

        return (MethodComponentContext) knnMethodContext.getMethodComponentContext().getParameters().get(METHOD_ENCODER_PARAMETER);
    }

    /**
     * Determine if the encoder parameter is specified
     *
     * @param knnMethodContext {@link KNNMethodContext}
     * @return true is the encoder is specified in the structure; false otherwise
     */
    protected boolean isEncoderSpecified(KNNMethodContext knnMethodContext) {
        return knnMethodContext != null
            && knnMethodContext.getMethodComponentContext().getParameters() != null
            && knnMethodContext.getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER);
    }

    protected boolean shouldEncoderBeResolved(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        // The encoder should not be resolved if:
        // 1. The encoder is specified
        // 2. The compression is x1
        // 3. The compression is not specified, the mode is not disk-based, and the index was created before
        // the x32-default flip (KNN_DEFAULT_COMPRESSION_FLIP_VERSION). Starting with that version, x32 is
        // the default for every new HNSW index, so the encoder must be resolved even when neither
        // compression nor an on-disk mode was requested.
        if (isEncoderSpecified(knnMethodContext)) {
            return false;
        }

        if (knnMethodConfigContext.getCompressionLevel() == CompressionLevel.x1) {
            return false;
        }

        Version version = knnMethodConfigContext.getVersionCreated();
        boolean is32xDefaultVersion = version != null && version.onOrAfter(KNN_DEFAULT_COMPRESSION_FLIP_VERSION);
        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel()) == false
            && Mode.ON_DISK != knnMethodConfigContext.getMode()
            && is32xDefaultVersion == false) {
            return false;
        }

        if (VectorDataType.FLOAT != knnMethodConfigContext.getVectorDataType()) {
            return false;
        }

        return true;
    }

    protected ValidationException validateNotTrainingContext(
        boolean shouldRequireTraining,
        KNNEngine knnEngine,
        ValidationException validationException
    ) {
        if (shouldRequireTraining) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError(
                String.format(Locale.ROOT, "Cannot use \"%s\" engine from training context", knnEngine.getName())
            );
        }

        return validationException;
    }

    protected ValidationException validateCompressionSupported(
        CompressionLevel compressionLevel,
        Set<CompressionLevel> supportedCompressionLevels,
        KNNEngine knnEngine,
        ValidationException validationException
    ) {
        if (CompressionLevel.isConfigured(compressionLevel) && supportedCompressionLevels.contains(compressionLevel) == false) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError(
                String.format(Locale.ROOT, "\"%s\" does not support \"%s\" compression", knnEngine.getName(), compressionLevel.getName())
            );
        }
        return validationException;
    }

    protected ValidationException validateCompressionNotx1WhenOnDisk(
        KNNMethodConfigContext knnMethodConfigContext,
        ValidationException validationException
    ) {
        if (knnMethodConfigContext.getCompressionLevel() == CompressionLevel.x1 && knnMethodConfigContext.getMode() == Mode.ON_DISK) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError(
                String.format(Locale.ROOT, "Cannot specify \"x1\" compression level when using \"%s\" mode", Mode.ON_DISK.getName())
            );
        }
        return validationException;
    }

    protected void validateCompressionConflicts(CompressionLevel originalCompressionLevel, CompressionLevel resolvedCompressionLevel) {
        if (CompressionLevel.isConfigured(originalCompressionLevel)
            && CompressionLevel.isConfigured(resolvedCompressionLevel)
            && resolvedCompressionLevel != originalCompressionLevel) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError("Cannot specify an encoder that conflicts with the provided compression level");
            throw validationException;
        }
    }

    /**
     * Resolves the default compression level from the config context. The resolution order is:
     * <ol>
     *   <li>If the compression level is explicitly configured, return it directly.</li>
     *   <li>If the index was created on or after {@code KNN_DEFAULT_COMPRESSION_FLIP_VERSION} (V_3_9_0), return
     *       x32 regardless of mode. This is the default-compression flip: every new HNSW index gets x32
     *       compression unless the user explicitly asked for something else.</li>
     *   <li>Otherwise, for ON_DISK mode on V_3_6_0+, return x32; for ON_DISK mode on earlier versions, return
     *       the provided fallback.</li>
     *   <li>Otherwise (older versions, non-ON_DISK mode), return x1.</li>
     * </ol>
     *
     * @param knnMethodConfigContext the config context
     * @param priorVersionOnDiskDefault the default compression for ON_DISK mode before V_3_6_0
     * @return the resolved default compression level
     */
    protected CompressionLevel getDefaultCompressionLevel(
        KNNMethodConfigContext knnMethodConfigContext,
        CompressionLevel priorVersionOnDiskDefault
    ) {
        if (CompressionLevel.isConfigured(knnMethodConfigContext.getCompressionLevel())) {
            return knnMethodConfigContext.getCompressionLevel();
        }
        Version version = knnMethodConfigContext.getVersionCreated();
        // The x32-default flip: on V_3_9_0+, x32 is the default for all new HNSW indices, regardless of mode.
        if (version != null && version.onOrAfter(KNN_DEFAULT_COMPRESSION_FLIP_VERSION)) {
            return CompressionLevel.x32;
        }
        if (knnMethodConfigContext.getMode() != Mode.ON_DISK) {
            return CompressionLevel.x1;
        }
        if (version != null && version.onOrAfter(Version.V_3_6_0)) {
            return CompressionLevel.x32;
        }
        return priorVersionOnDiskDefault;
    }
}
