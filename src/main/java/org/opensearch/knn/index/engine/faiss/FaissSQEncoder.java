/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_TYPES;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;

/**
 * Faiss SQ (Scalar Quantization) encoder.
 *
 * <p>Starting with 3.6.0, this encoder supports a {@code bits} parameter that controls the
 * quantization bit width:
 * <ul>
 *   <li>{@code bits=1} — 1-bit quantization, x32 compression. Uses a dedicated
 *       per-field format backed by Lucene's 1-bit scalar quantization. The {@code type} parameter
 *       is not allowed when bits=1.</li>
 *   <li>{@code bits=16} — equivalent to the existing {@code type=fp16} behavior, x2 compression</li>
 * </ul>
 *
 * <p>For indices created before 3.6.0, the encoder works as before with just the {@code type}
 * parameter (no {@code bits} required).
 *
 * <p>On 3.6.0+, {@code bits} is required when the encoder is explicitly specified.
 */
public class FaissSQEncoder implements Encoder {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);
    private static final Set<Integer> VALID_BITS = Arrays.stream(Bits.values()).map(Bits::getValue).collect(Collectors.toUnmodifiableSet());

    /**
     * Supported bit widths for SQ quantization. Each maps to a specific quantization strategy
     * and compression level.
     */
    @Getter
    @RequiredArgsConstructor
    public enum Bits {
        ONE(1, CompressionLevel.x32),
        SIXTEEN(16, CompressionLevel.x2);

        private final int value;
        private final CompressionLevel compressionLevel;

        public static Bits fromValue(int value) {
            for (Bits b : values()) {
                if (b.value == value) return b;
            }
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unsupported bits value: %d", value));
        }
    }

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            FAISS_SQ_TYPE,
            new Parameter.StringParameter(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16, (v, context) -> FAISS_SQ_ENCODER_TYPES.contains(v))
        )
        .addParameter(FAISS_SQ_CLIP, new Parameter.BooleanParameter(FAISS_SQ_CLIP, false, (v, context) -> Objects.nonNull(v)))
        .addParameter(SQ_BITS, new Parameter.IntegerParameter(SQ_BITS, null, (v, context) -> {
            if (v == null) {
                // bits is optional on pre-3.6.0 (legacy type-based path)
                return true;
            }
            return VALID_BITS.contains(v);
        }))
        .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
            Map<String, Object> params = methodComponentContext.getParameters();
            Object bitsObj = params.get(SQ_BITS);

            // bits=1 path: 1-bit quantization — use flat description, Faiss only builds the HNSW graph
            if (bitsObj instanceof Integer && (Integer) bitsObj == Bits.ONE.getValue()) {
                int bits = (Integer) bitsObj;
                return KNNLibraryIndexingContextImpl.builder().parameters(new HashMap<>() {
                    {
                        put(INDEX_DESCRIPTION_PARAMETER, FAISS_FLAT_DESCRIPTION);
                        put(NAME, ENCODER_SQ);
                        put(SQ_BITS, bits);
                    }
                }).build();
            }

            // Legacy/fp16 path: standard SQ description
            return MethodAsMapBuilder.builder(FAISS_SQ_DESCRIPTION, methodComponent, methodComponentContext, knnMethodConfigContext)
                .addParameter(FAISS_SQ_TYPE, "", "")
                .build();
        }))
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        if (methodComponentContext != null && methodComponentContext.getParameters().containsKey(SQ_BITS)) {
            Object bitsObj = methodComponentContext.getParameters().get(SQ_BITS);
            if (bitsObj instanceof Integer) {
                return Bits.fromValue((Integer) bitsObj).getCompressionLevel();
            }
        }
        // Legacy path — type=fp16 is x2
        return CompressionLevel.x2;
    }

    @Override
    public TrainingConfigValidationOutput validateEncoderConfig(TrainingConfigValidationInput validationInput) {
        TrainingConfigValidationOutput.TrainingConfigValidationOutputBuilder builder = TrainingConfigValidationOutput.builder();
        KNNMethodContext knnMethodContext = validationInput.getKnnMethodContext();
        KNNMethodConfigContext configContext = validationInput.getKnnMethodConfigContext();

        if (knnMethodContext == null || configContext == null) {
            return builder.build();
        }

        MethodComponentContext encoderContext = (MethodComponentContext) knnMethodContext.getMethodComponentContext()
            .getParameters()
            .get(METHOD_ENCODER_PARAMETER);
        if (encoderContext == null) {
            return builder.build();
        }

        Map<String, Object> encoderParams = encoderContext.getParameters();
        Version version = configContext.getVersionCreated();
        boolean isV360OrLater = version != null && version.onOrAfter(Version.V_3_6_0);
        Object bitsObj = encoderParams.get(SQ_BITS);
        boolean hasType = encoderParams.containsKey(FAISS_SQ_TYPE);

        // On 3.6.0+, bits is required when the user explicitly specifies the sq encoder for FLOAT data
        if (isV360OrLater && bitsObj == null && configContext.getVectorDataType() == VectorDataType.FLOAT) {
            return builder.valid(false)
                .errorMessage(
                    String.format(
                        Locale.ROOT,
                        "Parameter [%s] is required for encoder [%s] on indices created with version 3.6.0 or later. "
                            + "Supported values: %s",
                        SQ_BITS,
                        ENCODER_SQ,
                        VALID_BITS
                    )
                )
                .build();
        }

        if (bitsObj instanceof Integer) {
            int bits = (Integer) bitsObj;

            // bits=1 does not support the type parameter
            if (bits == Bits.ONE.getValue() && hasType) {
                return builder.valid(false)
                    .errorMessage(
                        String.format(
                            Locale.ROOT,
                            "Parameter [%s] is not supported when [%s=%d] for encoder [%s]. "
                                + "The 1-bit scalar quantization path does not use the type parameter.",
                            FAISS_SQ_TYPE,
                            SQ_BITS,
                            bits,
                            ENCODER_SQ
                        )
                    )
                    .build();
            }

            // Validate compression level compatibility if explicitly set
            CompressionLevel configuredCompression = configContext.getCompressionLevel();
            if (CompressionLevel.isConfigured(configuredCompression)) {
                CompressionLevel expectedCompression = Bits.fromValue(bits).getCompressionLevel();
                if (configuredCompression != expectedCompression) {
                    return builder.valid(false)
                        .errorMessage(
                            String.format(
                                Locale.ROOT,
                                "Compression level [%s] is incompatible with [%s=%d] for encoder [%s]. "
                                    + "Expected compression level: [%s]",
                                configuredCompression.getName(),
                                SQ_BITS,
                                bits,
                                ENCODER_SQ,
                                expectedCompression.getName()
                            )
                        )
                        .build();
                }
            }
        }

        return builder.build();
    }

    /**
     * Checks whether the given method parameters map contains an sq encoder with bits=1.
     * Works with both the method component parameters (from KNNMethodContext) and the
     * per-field params map (from codec format resolver).
     *
     * @param params map that may contain a {@code METHOD_ENCODER_PARAMETER} entry
     * @return true if the encoder is sq with bits=1
     */
    public static boolean isSQOneBit(Map<String, Object> params) {
        if (params == null) {
            return false;
        }
        Object encoderObj = params.get(METHOD_ENCODER_PARAMETER);
        if (encoderObj instanceof MethodComponentContext == false) {
            return false;
        }
        MethodComponentContext encoderCtx = (MethodComponentContext) encoderObj;
        if (ENCODER_SQ.equals(encoderCtx.getName()) == false) {
            return false;
        }
        Object bits = encoderCtx.getParameters().get(SQ_BITS);
        return bits instanceof Integer && (Integer) bits == Bits.ONE.getValue();
    }
}
