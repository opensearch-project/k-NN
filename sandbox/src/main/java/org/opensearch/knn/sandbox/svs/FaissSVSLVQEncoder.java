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
import org.opensearch.knn.sandbox.ExperimentalAlgorithm;

import java.util.Locale;
import java.util.Set;

import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_ENCODER_LVQ;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_PRIMARY_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.METHOD_PARAMETER_LVQ_RESIDUAL_BITS;

/**
 * LVQ (Locally-adaptive Vector Quantization) encoder for SVS indexes, with {@code primary_bits} and
 * {@code residual_bits} parameters (both default 4). Only the combinations the SVS runtime's
 * {@code SVSStorageKind} supports — {@code (4,0)}, {@code (4,4)}, {@code (4,8)} — are accepted; others are
 * rejected at index creation rather than failing deep in native code.
 */
@ExperimentalAlgorithm(description = "Intel SVS Vamana LVQ encoder", since = "3.8.0")
public class FaissSVSLVQEncoder implements Encoder {

    static final int DEFAULT_PRIMARY_BITS = 4;
    static final int DEFAULT_RESIDUAL_BITS = 4;

    /** Supported (primary_bits, residual_bits) combinations, mirroring the SVS {@code SVSStorageKind} LVQ kinds. */
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
            // Builds the "LVQ{primary}x{residual}" token, e.g. "LVQ4x4".
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

    private static void validatePlatformSupportsLvq() {
        if (SvsService.isLvqLeanvecEnabled() == false) {
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
        // Map 32/total_bits to the nearest supported CompressionLevel; 4x8 (~2.67x) has no exact enum, use x4.
        if (totalBits <= 4) {
            return CompressionLevel.x8;
        }
        return CompressionLevel.x4;
    }
}
