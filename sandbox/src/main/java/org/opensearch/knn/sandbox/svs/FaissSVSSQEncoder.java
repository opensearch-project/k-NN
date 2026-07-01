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

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_FP16_DESCRIPTION;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_SQ8_DESCRIPTION;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE_FP16;
import static org.opensearch.knn.sandbox.svs.SVSConstants.FAISS_SVS_SQ_TYPE_SQ8;

/**
 * Scalar-quantization encoder for SVS Vamana, exposed as the unified {@code sq} encoder (matching the HNSW
 * convention) with a {@code type} parameter: {@code fp16} (default, x2) or {@code sq8} (x4). Unlike the main
 * {@link org.opensearch.knn.index.engine.faiss.FaissSQEncoder} it has no Lucene {@code bits} path; SVS
 * quantization is performed natively.
 */
@ExperimentalAlgorithm(description = "Intel SVS Vamana scalar-quantization (sq) encoder", since = "3.8.0")
public class FaissSVSSQEncoder implements Encoder {

    private static final Set<String> SUPPORTED_TYPES = Set.of(FAISS_SVS_SQ_TYPE_FP16, FAISS_SVS_SQ_TYPE_SQ8);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
        .addParameter(
            FAISS_SVS_SQ_TYPE,
            new Parameter.StringParameter(FAISS_SVS_SQ_TYPE, FAISS_SVS_SQ_TYPE_FP16, (v, context) -> SUPPORTED_TYPES.contains(v))
        )
        // 'bits' is the HNSW sq knob, not ours. Declared (null default) only so a user who passes it gets the
        // targeted message from validateNoBitsParameter, not a generic "unknown parameter" rejection.
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

    /**
     * Rejects the HNSW sq encoder's {@code bits} parameter with a targeted message; SVS sq selects precision
     * via {@code type} and has no bit-width knob.
     */
    static void validateNoBitsParameter(MethodComponentContext methodComponentContext) {
        if (methodComponentContext == null) {
            return;
        }
        // Only a user-supplied (non-null) bits triggers the message, not the null default from the whitelist.
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
}
