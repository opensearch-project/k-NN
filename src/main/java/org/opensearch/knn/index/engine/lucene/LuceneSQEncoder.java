/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.Parameter;

import java.util.List;

import static org.opensearch.knn.common.KNNConstants.DYNAMIC_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.MAXIMUM_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.MINIMUM_CONFIDENCE_INTERVAL;

/**
 * Lucene scalar quantization encoder
 */
public class LuceneSQEncoder implements Encoder {
    private final static List<Integer> LUCENE_SQ_BITS_SUPPORTED = List.of(7);
    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addParameter(
            LUCENE_SQ_CONFIDENCE_INTERVAL,
            new Parameter.DoubleParameter(
                LUCENE_SQ_CONFIDENCE_INTERVAL,
                null,
                v -> v == DYNAMIC_CONFIDENCE_INTERVAL || (v >= MINIMUM_CONFIDENCE_INTERVAL && v <= MAXIMUM_CONFIDENCE_INTERVAL)
            )
        )
        .addParameter(
            LUCENE_SQ_BITS,
            new Parameter.IntegerParameter(LUCENE_SQ_BITS, LUCENE_SQ_DEFAULT_BITS, LUCENE_SQ_BITS_SUPPORTED::contains)
        )
        .build();

    @Override
    public String getName() {
        return METHOD_COMPONENT.getName();
    }

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }
}
