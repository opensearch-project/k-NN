/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.*;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.*;

/**
 * Optimized Scalar Quantizer encoder
 */
public class OptimizedScalarQuantizerEncoder implements Encoder {
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    private final static List<Integer> OPTIMIZED_SCALAR_QUANTIZER_BITS_SUPPORTED = List.of(1);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_OPTIMIZED_SCALAR_QUANTIZER)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            OPTIMIZED_SCALAR_QUANTIZER_BITS,
            new Parameter.IntegerParameter(
                OPTIMIZED_SCALAR_QUANTIZER_BITS,
                OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS,
                (v, context) -> OPTIMIZED_SCALAR_QUANTIZER_BITS_SUPPORTED.contains(v)
            )
        )
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
        return CompressionLevel.x32;
    }
}
