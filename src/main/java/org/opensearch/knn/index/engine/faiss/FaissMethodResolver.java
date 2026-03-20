/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

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
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
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

        // TODO: This chain of if-blocks mapping compression levels to encoder configs is too complex.
        // Need to refactor it into a strategy or registry pattern where each CompressionLevel declares
        // its own encoder factory, e.g. CompressionLevel.x2.createEncoder(context, encoderMap). That
        // would make it easier to add new compression level resolutions.
        MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_FLAT, new HashMap<>());
        Encoder encoder = encoderMap.get(ENCODER_FLAT);
        if (CompressionLevel.x2 == resolvedCompressionLevel) {
            encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
            encoder = encoderMap.get(ENCODER_SQ);
            encoderComponentContext.getParameters().put(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16);
            // On 3.6.0+, also set bits for consistency with the new bits-based validation
            if (knnMethodConfigContext.getVersionCreated() != null
                && knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)) {
                encoderComponentContext.getParameters().put(SQ_BITS, FaissSQEncoder.Bits.SIXTEEN.getValue());
            }
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
            if (shouldUseSQOneBitForX32(knnMethodConfigContext, encoderMap)) {
                encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
                encoder = encoderMap.get(ENCODER_SQ);
                encoderComponentContext.getParameters().put(SQ_BITS, FaissSQEncoder.Bits.ONE.getValue());
            } else {
                encoderComponentContext = new MethodComponentContext(QFrameBitEncoder.NAME, new HashMap<>());
                encoder = encoderMap.get(QFrameBitEncoder.NAME);
                encoderComponentContext.getParameters().put(QFrameBitEncoder.BITCOUNT_PARAM, CompressionLevel.x32.numBitsForFloat32());
            }
        }

        Map<String, Object> resolvedParams = MethodComponent.getParameterMapWithDefaultsAdded(
            encoderComponentContext,
            encoder.getMethodComponent(),
            knnMethodConfigContext
        );
        encoderComponentContext.getParameters().putAll(resolvedParams);

        // When auto-resolved to bits=1, remove the type default that was injected — the
        // 1-bit quantization path doesn't use it, and validateEncoderConfig would reject it.
        if (encoderComponentContext.getParameters().get(SQ_BITS) instanceof Integer bitsVal
            && bitsVal == FaissSQEncoder.Bits.ONE.getValue()) {
            encoderComponentContext.getParameters().remove(FAISS_SQ_TYPE);
        }

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

    /**
     * Starting 3.6.0, x32 compression can use sq(bits=1) instead of the older QFrameBitEncoder (binary).
     * 1-bit quantization delegates to Lucene's flat format rather than the k-NN quantization
     * framework, which gives better recall. The encoderMap guard is needed because IVF doesn't
     * register the sq encoder — only HNSW does.
     *
     * Currently disabled — the BBQ writer pipeline is not yet fully stable for auto-resolved
     * indices. Users can still explicitly specify sq(bits=1) to opt in. This will be enabled
     * as the default in Part 2.
     * TODO: Enable once the Faiss1040ScalarQuantizedKnnVectorsWriter pipeline is validated end-to-end.
     */
    private static boolean shouldUseSQOneBitForX32(KNNMethodConfigContext knnMethodConfigContext, Map<String, Encoder> encoderMap) {
        return false;
    }
}
