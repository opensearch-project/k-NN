/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_PRIMARY_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_RESIDUAL_BITS;

/**
 * LVQ encoder for SVS indexes: {@code primary_bits} x {@code residual_bits}, supported 4x0, 4x4, 4x8.
 */
public class FaissSVSLVQEncoder implements Encoder {

    static final int DEFAULT_PRIMARY_BITS = 4;
    static final int DEFAULT_RESIDUAL_BITS = 4;

    private static final Set<String> SUPPORTED_BIT_COMBINATIONS = Set.of("4x0", "4x4", "4x8");

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(FAISS_SVS_ENCODER_LVQ)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .addParameter(
            METHOD_PARAMETER_LVQ_PRIMARY_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LVQ_PRIMARY_BITS, DEFAULT_PRIMARY_BITS, (v, context) -> v >= 1 && v <= 8)
        )
        .addParameter(
            METHOD_PARAMETER_LVQ_RESIDUAL_BITS,
            new Parameter.IntegerParameter(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, DEFAULT_RESIDUAL_BITS, (v, context) -> v >= 0 && v <= 8)
        )
        .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
            int primaryBits = readBits(methodComponentContext, METHOD_PARAMETER_LVQ_PRIMARY_BITS, DEFAULT_PRIMARY_BITS);
            int residualBits = readBits(methodComponentContext, METHOD_PARAMETER_LVQ_RESIDUAL_BITS, DEFAULT_RESIDUAL_BITS);
            validateBitCombination(primaryBits, residualBits);
            validatePlatformSupportsLvq();

            SvsMethodAsMapBuilder builder = SvsMethodAsMapBuilder.builder(
                "LVQ",
                methodComponent,
                methodComponentContext,
                knnMethodConfigContext
            );
            builder.addParameter(METHOD_PARAMETER_LVQ_PRIMARY_BITS, "", "x");
            builder.addParameter(METHOD_PARAMETER_LVQ_RESIDUAL_BITS, "", "");
            return builder.build();
        }))
        .build();

    private static int readBits(MethodComponentContext methodComponentContext, String name, int defaultValue) {
        if (methodComponentContext == null) {
            return defaultValue;
        }
        Object value = methodComponentContext.getParameters().get(name);
        return value instanceof Integer ? (Integer) value : defaultValue;
    }

    static void validateBitCombination(int primaryBits, int residualBits) {
        String combination = String.format(Locale.ROOT, "%dx%d", primaryBits, residualBits);
        if (SUPPORTED_BIT_COMBINATIONS.contains(combination) == false) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Unsupported LVQ (primary_bits, residual_bits) combination [%s] for encoder [%s]. "
                        + "Supported combinations are: 4x0, 4x4, 4x8.",
                    combination,
                    FAISS_SVS_ENCODER_LVQ
                )
            );
        }
    }

    static void validatePlatformSupportsLvq() {
        final boolean lvqEnabled;
        try {
            lvqEnabled = SvsService.isLvqLeanvecEnabled();
        } catch (UnsatisfiedLinkError | ExceptionInInitializerError | NoClassDefFoundError e) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Encoder [%s] is not supported on this node: the SVS native library could not be loaded (%s)",
                    FAISS_SVS_ENCODER_LVQ,
                    e.getClass().getSimpleName()
                ),
                e
            );
        }
        if (lvqEnabled == false) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Encoder [%s] is not supported on this node. LVQ compression requires Intel SIMD support "
                        + "in the SVS runtime, which is unavailable on this platform or build.",
                    FAISS_SVS_ENCODER_LVQ
                )
            );
        }
    }

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext encoderContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        int primaryBits = readBits(encoderContext, METHOD_PARAMETER_LVQ_PRIMARY_BITS, DEFAULT_PRIMARY_BITS);
        int residualBits = readBits(encoderContext, METHOD_PARAMETER_LVQ_RESIDUAL_BITS, DEFAULT_RESIDUAL_BITS);
        int totalBits = primaryBits + residualBits;
        // 4x8 (~2.67x) has no exact level; report x4.
        if (totalBits <= 4) {
            return CompressionLevel.x8;
        }
        return CompressionLevel.x4;
    }

    @Override
    public Set<QuantizationBits> getSupportedBits() {
        return EnumSet.of(QuantizationBits.FOUR, QuantizationBits.SEVEN);
    }

    @Override
    public EncoderType getEncoderType() {
        // Closest fit in the closed enum.
        return EncoderType.SQ;
    }

}
