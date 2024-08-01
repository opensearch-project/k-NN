/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.DYNAMIC_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.MAXIMUM_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.MINIMUM_CONFIDENCE_INTERVAL;

/**
 * Lucene HNSW implementation
 */
public class LuceneHNSWMethod extends AbstractKNNMethod {

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.L2,
        SpaceType.COSINESIMIL,
        SpaceType.INNER_PRODUCT
    );

    /**
     * Constructor for LuceneHNSWMethod
     *
     * @see AbstractKNNMethod
     */
    public LuceneHNSWMethod() {
        super(initMethodComponent(), Set.copyOf(SUPPORTED_SPACES), new LuceneHNSWContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_HNSW)
            .addParameter(
                METHOD_PARAMETER_M,
                new Parameter.IntegerParameter(METHOD_PARAMETER_M, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, v -> v > 0)
            )
            .addParameter(
                METHOD_PARAMETER_EF_CONSTRUCTION,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                    v -> v > 0
                )
            )
            .addParameter(METHOD_ENCODER_PARAMETER, initEncoderParameter())
            .build();
    }

    private static Parameter.MethodComponentContextParameter initEncoderParameter() {
        List<Integer> LUCENE_SQ_BITS_SUPPORTED = List.of(7);
        Map<String, MethodComponent> supportedEncoders = ImmutableMap.of(
            ENCODER_SQ,
            MethodComponent.Builder.builder(ENCODER_SQ)
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
                .build()
        );

        MethodComponentContext defaultEncoder = new MethodComponentContext(KNNConstants.ENCODER_FLAT, Collections.emptyMap());

        return new Parameter.MethodComponentContextParameter(METHOD_ENCODER_PARAMETER, defaultEncoder, supportedEncoders);
    }
}
