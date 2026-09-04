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

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_FP16_DESCRIPTION;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_SQ8_DESCRIPTION;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE_FP16;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE_SQ8;

/**
 * {@code sq} encoder for SVS Vamana: {@code type} is {@code fp16} (default, x2) or {@code sq8} (x4).
 */
public class FaissSVSSQEncoder implements Encoder {

    private static final Set<String> SUPPORTED_TYPES = Set.of(FAISS_SVS_SQ_TYPE_FP16, FAISS_SVS_SQ_TYPE_SQ8);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .addParameter(
            FAISS_SVS_SQ_TYPE,
            new Parameter.StringParameter(FAISS_SVS_SQ_TYPE, FAISS_SVS_SQ_TYPE_FP16, (v, context) -> SUPPORTED_TYPES.contains(v))
        )
        // Declared only so a user passing the HNSW sq 'bits' knob gets a targeted message.
        .addParameter(SQ_BITS, new Parameter.IntegerParameter(SQ_BITS, null, (v, context) -> true))
        .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
            validateNoBitsParameter(methodComponentContext);
            String description = FAISS_SVS_SQ_FP16_DESCRIPTION;
            if (FAISS_SVS_SQ_TYPE_SQ8.equals(resolveType(methodComponentContext))) {
                description = FAISS_SVS_SQ_SQ8_DESCRIPTION;
            }
            return SvsMethodAsMapBuilder.builder(description, methodComponent, methodComponentContext, knnMethodConfigContext).build();
        }))
        .build();

    static void validateNoBitsParameter(MethodComponentContext methodComponentContext) {
        if (methodComponentContext == null) {
            return;
        }
        if (methodComponentContext.getParameters().get(SQ_BITS) != null) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "The svs_vamana '%s' encoder uses the '%s' parameter (%s|%s), not '%s'.",
                    ENCODER_SQ,
                    FAISS_SVS_SQ_TYPE,
                    FAISS_SVS_SQ_TYPE_FP16,
                    FAISS_SVS_SQ_TYPE_SQ8,
                    SQ_BITS
                )
            );
        }
    }

    private static String resolveType(MethodComponentContext methodComponentContext) {
        if (methodComponentContext == null) {
            return FAISS_SVS_SQ_TYPE_FP16;
        }
        Object type = methodComponentContext.getParameters().get(FAISS_SVS_SQ_TYPE);
        return type instanceof String ? (String) type : FAISS_SVS_SQ_TYPE_FP16;
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
        return FAISS_SVS_SQ_TYPE_SQ8.equals(resolveType(encoderContext)) ? CompressionLevel.x4 : CompressionLevel.x2;
    }

    @Override
    public Set<QuantizationBits> getSupportedBits() {
        return EnumSet.of(QuantizationBits.SEVEN, QuantizationBits.SIXTEEN);
    }

    @Override
    public EncoderType getEncoderType() {
        return EncoderType.SQ;
    }

}
