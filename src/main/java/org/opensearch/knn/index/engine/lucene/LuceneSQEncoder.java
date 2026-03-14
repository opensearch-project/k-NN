/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;
import org.opensearch.common.logging.DeprecationLogger;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.DYNAMIC_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.CONFIDENCE_INTERVAL_DEPRECATION_MSG;
import static org.opensearch.knn.common.KNNConstants.MAXIMUM_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.MINIMUM_CONFIDENCE_INTERVAL;

/**
 * Lucene scalar quantization encoder
 */
public class LuceneSQEncoder implements Encoder {
    private static final DeprecationLogger deprecationLogger = DeprecationLogger.getLogger(LuceneSQEncoder.class);
    private static final String CONFIDENCE_INTERVAL_DEPRECATION_KEY = "confidence_interval_deprecated";
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);
    private final static List<Integer> LUCENE_SQ_BITS_SUPPORTED = List.of(7);
    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(LUCENE_SQ_CONFIDENCE_INTERVAL, new Parameter.DoubleParameter(LUCENE_SQ_CONFIDENCE_INTERVAL, null, (v, context) -> {
            deprecationLogger.deprecate(
                CONFIDENCE_INTERVAL_DEPRECATION_KEY,
                CONFIDENCE_INTERVAL_DEPRECATION_MSG,
                LUCENE_SQ_CONFIDENCE_INTERVAL,
                ENCODER_SQ
            );
            return v == DYNAMIC_CONFIDENCE_INTERVAL || (v >= MINIMUM_CONFIDENCE_INTERVAL && v <= MAXIMUM_CONFIDENCE_INTERVAL);
        }))
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
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        // Hard coding to 4x for now, given thats all that is supported.
        return CompressionLevel.x4;
    }
}
