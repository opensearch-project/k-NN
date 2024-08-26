/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultHnswSearchContext;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.config.CompressionConfig;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.FAISS_HNSW_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

/**
 * Faiss HNSW method implementation
 */
public class FaissHNSWMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(
        VectorDataType.FLOAT,
        VectorDataType.BINARY,
        VectorDataType.BYTE
    );

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(SpaceType.HAMMING, SpaceType.L2, SpaceType.INNER_PRODUCT);

    private final static MethodComponentContext DEFAULT_ENCODER_CONTEXT = new MethodComponentContext(
        KNNConstants.ENCODER_FLAT,
        Collections.emptyMap()
    );
    private final static MethodComponentContext DEFAULT_32x_ENCODER_CONTEXT = new MethodComponentContext(
        QFrameBitEncoder.NAME,
        Map.of(QFrameBitEncoder.BITCOUNT_PARAM, 1)
    );
    private final static MethodComponentContext DEFAULT_16x_ENCODER_CONTEXT = new MethodComponentContext(
        QFrameBitEncoder.NAME,
        Map.of(QFrameBitEncoder.BITCOUNT_PARAM, 2)
    );
    private final static MethodComponentContext DEFAULT_8x_ENCODER_CONTEXT = new MethodComponentContext(
        QFrameBitEncoder.NAME,
        Map.of(QFrameBitEncoder.BITCOUNT_PARAM, 4)
    );

    private final static List<Encoder> SUPPORTED_ENCODERS = List.of(
        new FaissFlatEncoder(),
        new FaissSQEncoder(),
        new FaissHNSWPQEncoder(),
        new QFrameBitEncoder()
    );

    /**
     * Constructor for FaissHNSWMethod
     *
     * @see AbstractKNNMethod
     */
    public FaissHNSWMethod() {
        super(initMethodComponent(), Set.copyOf(SUPPORTED_SPACES), new DefaultHnswSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_HNSW)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_M,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_M,
                    knnMethodConfigContext -> KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                METHOD_PARAMETER_EF_CONSTRUCTION,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    knnMethodConfigContext -> KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                METHOD_PARAMETER_EF_SEARCH,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_EF_SEARCH,
                    knnMethodConfigContext -> KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                    (v, context) -> v > 0
                )
            )
            .addParameter(METHOD_ENCODER_PARAMETER, initEncoderParameter())
            .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                MethodAsMapBuilder methodAsMapBuilder = MethodAsMapBuilder.builder(
                    FAISS_HNSW_DESCRIPTION,
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                ).addParameter(METHOD_PARAMETER_M, "", "").addParameter(METHOD_ENCODER_PARAMETER, ",", "");
                return adjustIndexDescription(methodAsMapBuilder, methodComponentContext, knnMethodConfigContext);
            }))
            .build();
    }

    private static Parameter.MethodComponentContextParameter initEncoderParameter() {
        return new Parameter.MethodComponentContextParameter(METHOD_ENCODER_PARAMETER, knnMethodConfigContext -> {
            CompressionConfig compressionConfig = resolveCompressionConfig(knnMethodConfigContext);
            if (compressionConfig == CompressionConfig.NOT_CONFIGURED) {
                return DEFAULT_ENCODER_CONTEXT;
            }

            return getDefaultEncoderFromCompression(compressionConfig);

        }, SUPPORTED_ENCODERS.stream().collect(Collectors.toMap(Encoder::getName, Encoder::getMethodComponent)));
    }

    private static MethodComponentContext getDefaultEncoderFromCompression(CompressionConfig compressionConfig) {
        if (compressionConfig == CompressionConfig.x32) {
            return DEFAULT_32x_ENCODER_CONTEXT;
        }

        if (compressionConfig == CompressionConfig.x16) {
            return DEFAULT_16x_ENCODER_CONTEXT;
        }

        if (compressionConfig == CompressionConfig.x8) {
            return DEFAULT_8x_ENCODER_CONTEXT;
        }

        return DEFAULT_ENCODER_CONTEXT;
    }

}
