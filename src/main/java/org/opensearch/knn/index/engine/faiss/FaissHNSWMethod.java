/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultHnswContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.FAISS_HNSW_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_PQ_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.index.engine.faiss.Common.COMMON_ENCODERS;

/**
 * Faiss HNSW method implementation
 */
public class FaissHNSWMethod extends AbstractKNNMethod {
    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.HAMMING,
        SpaceType.L2,
        SpaceType.INNER_PRODUCT
    );

    /**
     * Constructor for FaissHNSWMethod
     *
     * @see AbstractKNNMethod
     */
    public FaissHNSWMethod() {
        super(initMethodComponent(), Set.copyOf(SUPPORTED_SPACES), new DefaultHnswContext());
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
            .addParameter(
                METHOD_PARAMETER_EF_SEARCH,
                new Parameter.IntegerParameter(METHOD_PARAMETER_EF_SEARCH, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH, v -> v > 0)
            )
            .addParameter(METHOD_ENCODER_PARAMETER, initEncoderParameter())
            .setMapGenerator(
                ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                    FAISS_HNSW_DESCRIPTION,
                    methodComponent,
                    methodComponentContext
                ).addParameter(METHOD_PARAMETER_M, "", "").addParameter(METHOD_ENCODER_PARAMETER, ",", "").build())
            )
            .build();
    }

    private static Parameter.MethodComponentContextParameter initEncoderParameter() {
        Map<String, MethodComponent> supportedEncoders = ImmutableMap.<String, MethodComponent>builder()
            .putAll(
                ImmutableMap.of(
                    KNNConstants.ENCODER_PQ,
                    MethodComponent.Builder.builder(KNNConstants.ENCODER_PQ)
                        .addParameter(
                            ENCODER_PARAMETER_PQ_M,
                            new Parameter.IntegerParameter(
                                ENCODER_PARAMETER_PQ_M,
                                ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT,
                                v -> v > 0 && v < ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT,
                                (v, vectorSpaceInfo) -> vectorSpaceInfo.getDimension() % v == 0
                            )
                        )
                        .addParameter(
                            ENCODER_PARAMETER_PQ_CODE_SIZE,
                            new Parameter.IntegerParameter(
                                ENCODER_PARAMETER_PQ_CODE_SIZE,
                                ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT,
                                v -> Objects.equals(v, ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT)
                            )
                        )
                        .setRequiresTraining(true)
                        .setMapGenerator(
                            ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                                FAISS_PQ_DESCRIPTION,
                                methodComponent,
                                methodComponentContext
                            ).addParameter(ENCODER_PARAMETER_PQ_M, "", "").build())
                        )
                        .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
                            int codeSize = ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
                            return ((4L * (1L << codeSize) * dimension) / BYTES_PER_KILOBYTES) + 1;
                        })
                        .build()
                )
            )
            .putAll(COMMON_ENCODERS)
            .build();

        MethodComponentContext defaultEncoder = new MethodComponentContext(KNNConstants.ENCODER_FLAT, Collections.emptyMap());

        return new Parameter.MethodComponentContextParameter(METHOD_ENCODER_PARAMETER, defaultEncoder, supportedEncoders);
    }
}
