/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DYNAMIC_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.MAXIMUM_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MINIMUM_CONFIDENCE_INTERVAL;

/**
 * Lucene scalar quantization encoder
 */
public class LuceneSQEncoder implements Encoder {
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);
    private static final Set<Integer> LUCENE_SQ_BITS_SUPPORTED = Arrays.stream(Bits.values())
        .map(Bits::getValue)
        .collect(Collectors.toUnmodifiableSet());
    private static final Bits LUCENE_PRE_360_SUPPORTED_SQ_BITS = Bits.SEVEN;

    /**
     * Supported bit widths for SQ quantization. Each maps to a specific quantization strategy
     * and compression level.
     */
    @Getter
    @RequiredArgsConstructor
    public enum Bits {
        ONE(1, CompressionLevel.x32),
        SEVEN(7, CompressionLevel.x4);

        private final int value;
        private final CompressionLevel compressionLevel;

        public static Bits fromValue(int value) {
            for (Bits b : values()) {
                if (b.value == value) return b;
            }
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Unsupported bits value: %d", value));
        }
    }

    // Lucene SQ supports compression to 1 bit only in indices with version >= 3.6.0
    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            LUCENE_SQ_CONFIDENCE_INTERVAL,
            new Parameter.DoubleParameter(
                LUCENE_SQ_CONFIDENCE_INTERVAL,
                null,
                (v, context) -> v == DYNAMIC_CONFIDENCE_INTERVAL || (v >= MINIMUM_CONFIDENCE_INTERVAL && v <= MAXIMUM_CONFIDENCE_INTERVAL)
            )
        )
        .addParameter(
            LUCENE_SQ_BITS,
            new Parameter.IntegerParameter(LUCENE_SQ_BITS, LUCENE_SQ_DEFAULT_BITS, (v, context) -> LUCENE_SQ_BITS_SUPPORTED.contains(v))
        )
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
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
        Object bitsObj = encoderParams.get(LUCENE_SQ_BITS);
        Set<String> nonBitParameters = encoderParams.keySet().stream().filter(k -> !k.equals(LUCENE_SQ_BITS)).collect(Collectors.toSet());

        // On 3.6.0+, bits is required when the user explicitly specifies the lucene sq encoder
        if (isV360OrLater && bitsObj == null) {
            return builder.valid(false)
                .errorMessage(
                    String.format(
                        Locale.ROOT,
                        "Parameter [%s] is required for encoder [%s] on indices created with version 3.6.0 or later. "
                            + "Supported values: %s",
                        LUCENE_SQ_BITS,
                        ENCODER_SQ,
                        LUCENE_SQ_BITS_SUPPORTED
                    )
                )
                .build();
        }

        if (bitsObj instanceof Integer) {
            int bits = (Integer) bitsObj;

            // bits=1 does not support other parameters
            if (bits == Bits.ONE.getValue()) {
                if (!nonBitParameters.isEmpty()) {
                    return builder.valid(false)
                        .errorMessage(
                            String.format(
                                Locale.ROOT,
                                "Parameters [%s] are not supported when [%s=%d] for encoder [%s]. "
                                    + "The 1-bit scalar quantization path does not use the type parameter.",
                                nonBitParameters,
                                LUCENE_SQ_BITS,
                                bits,
                                ENCODER_SQ
                            )
                        )
                        .build();
                }
                if (!isV360OrLater) {
                    return builder.valid(false)
                        .errorMessage(
                            String.format(
                                Locale.ROOT,
                                "Parameter [%s=%d] is only supported for indices created with version 3.6.0 or later. "
                                    + "Supported values: %s",
                                LUCENE_SQ_BITS,
                                bits,
                                LUCENE_PRE_360_SUPPORTED_SQ_BITS
                            )
                        )
                        .build();
                }
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
                                LUCENE_SQ_BITS,
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

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        if (knnMethodConfigContext == null) {
            return CompressionLevel.x4;
        }

        if (knnMethodConfigContext.getCompressionLevel() == CompressionLevel.NOT_CONFIGURED) {
            if (knnMethodConfigContext.getVersionCreated().onOrAfter(Version.V_3_6_0)) {
                return CompressionLevel.x32;
            }
            return CompressionLevel.x4;
        }
        return knnMethodConfigContext.getCompressionLevel();
    }
}
