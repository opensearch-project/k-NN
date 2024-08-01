/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultIVFContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_COUNT_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.FAISS_IVF_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_PQ_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_LIMIT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_LIMIT;
import static org.opensearch.knn.index.engine.faiss.Common.COMMON_ENCODERS;

/**
 * Faiss ivf implementation
 */
public class FaissIVFMethod extends AbstractKNNMethod {

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.L2,
        SpaceType.INNER_PRODUCT,
        SpaceType.HAMMING
    );

    /**
     * Constructor for FaissIVFMethod
     *
     * @see AbstractKNNMethod
     */
    public FaissIVFMethod() {
        super(initMethodComponent(), Set.copyOf(SUPPORTED_SPACES), new DefaultIVFContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_IVF)
            .addParameter(
                METHOD_PARAMETER_NPROBES,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_NPROBES,
                    METHOD_PARAMETER_NPROBES_DEFAULT,
                    v -> v > 0 && v < METHOD_PARAMETER_NPROBES_LIMIT
                )
            )
            .addParameter(
                METHOD_PARAMETER_NLIST,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_NLIST,
                    METHOD_PARAMETER_NLIST_DEFAULT,
                    v -> v > 0 && v < METHOD_PARAMETER_NLIST_LIMIT
                )
            )
            .addParameter(METHOD_ENCODER_PARAMETER, initEncoderParameter())
            .setRequiresTraining(true)
            .setMapGenerator(
                ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                    FAISS_IVF_DESCRIPTION,
                    methodComponent,
                    methodComponentContext
                ).addParameter(METHOD_PARAMETER_NLIST, "", "").addParameter(METHOD_ENCODER_PARAMETER, ",", "").build())
            )
            .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
                // Size estimate formula: (4 * nlists * d) / 1024 + 1

                // Get value of nlists passed in by user
                Object nlistObject = methodComponentContext.getParameters().get(METHOD_PARAMETER_NLIST);

                // If not specified, get default value of nlist
                if (nlistObject == null) {
                    Parameter<?> nlistParameter = methodComponent.getParameters().get(METHOD_PARAMETER_NLIST);
                    if (nlistParameter == null) {
                        throw new IllegalStateException(
                            String.format("%s  is not a valid parameter. This is a bug.", METHOD_PARAMETER_NLIST)
                        );
                    }

                    nlistObject = nlistParameter.getDefaultValue();
                }

                if (!(nlistObject instanceof Integer)) {
                    throw new IllegalStateException(String.format("%s must be an integer.", METHOD_PARAMETER_NLIST));
                }

                int centroids = (Integer) nlistObject;
                return ((4L * centroids * dimension) / BYTES_PER_KILOBYTES) + 1;
            })
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
                                v -> v > 0 && v < ENCODER_PARAMETER_PQ_CODE_SIZE_LIMIT
                            )
                        )
                        .setRequiresTraining(true)
                        .setMapGenerator(
                            ((methodComponent, methodComponentContext) -> MethodAsMapBuilder.builder(
                                FAISS_PQ_DESCRIPTION,
                                methodComponent,
                                methodComponentContext
                            ).addParameter(ENCODER_PARAMETER_PQ_M, "", "").addParameter(ENCODER_PARAMETER_PQ_CODE_SIZE, "x", "").build())
                        )
                        .setOverheadInKBEstimator((methodComponent, methodComponentContext, dimension) -> {
                            // Size estimate formula: (4 * d * 2^code_size) / 1024 + 1

                            // Get value of code size passed in by user
                            Object codeSizeObject = methodComponentContext.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);

                            // If not specified, get default value of code size
                            if (codeSizeObject == null) {
                                Parameter<?> codeSizeParameter = methodComponent.getParameters().get(ENCODER_PARAMETER_PQ_CODE_SIZE);
                                if (codeSizeParameter == null) {
                                    throw new IllegalStateException(
                                        String.format("%s  is not a valid parameter. This is a bug.", ENCODER_PARAMETER_PQ_CODE_SIZE)
                                    );
                                }

                                codeSizeObject = codeSizeParameter.getDefaultValue();
                            }

                            if (!(codeSizeObject instanceof Integer)) {
                                throw new IllegalStateException(String.format("%s must be an integer.", ENCODER_PARAMETER_PQ_CODE_SIZE));
                            }

                            int codeSize = (Integer) codeSizeObject;
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
