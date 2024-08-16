/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultIVFSearchContext;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.FAISS_IVF_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_LIMIT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES_LIMIT;
import static org.opensearch.knn.index.engine.faiss.Faiss.FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;

/**
 * Faiss ivf implementation
 */
public class FaissIVFMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT, VectorDataType.BINARY);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.L2,
        SpaceType.INNER_PRODUCT,
        SpaceType.HAMMING
    );

    private final static MethodComponentContext DEFAULT_ENCODER_CONTEXT = new MethodComponentContext(
        KNNConstants.ENCODER_FLAT,
        Collections.emptyMap()
    );
    private final static List<Encoder> SUPPORTED_ENCODERS = List.of(new FaissFlatEncoder(), new FaissSQEncoder(), new FaissIVFPQEncoder());

    /**
     * Constructor for FaissIVFMethod
     *
     * @see AbstractKNNMethod
     */
    public FaissIVFMethod() {
        super(initMethodComponent(), Set.copyOf(SUPPORTED_SPACES), new DefaultIVFSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_IVF)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_NPROBES,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_NPROBES,
                    METHOD_PARAMETER_NPROBES_DEFAULT,
                    (v, context) -> v > 0 && v < METHOD_PARAMETER_NPROBES_LIMIT
                )
            )
            .addParameter(
                METHOD_PARAMETER_NLIST,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_NLIST,
                    METHOD_PARAMETER_NLIST_DEFAULT,
                    (v, context) -> v > 0 && v < METHOD_PARAMETER_NLIST_LIMIT
                )
            )
            .addParameter(METHOD_ENCODER_PARAMETER, initEncoderParameter())
            .setRequiresTraining(true)
            .setMapGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
                String prefix = "";
                if (knnMethodConfigContext.getVectorDataType() == VectorDataType.BINARY) {
                    prefix = FAISS_BINARY_INDEX_DESCRIPTION_PREFIX;
                }

                return MethodAsMapBuilder.builder(
                    prefix + FAISS_IVF_DESCRIPTION,
                    methodComponent,
                    methodComponentContext,
                    knnMethodConfigContext
                ).addParameter(METHOD_PARAMETER_NLIST, "", "").addParameter(METHOD_ENCODER_PARAMETER, ",", "").build();
            }))
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
        return new Parameter.MethodComponentContextParameter(
            METHOD_ENCODER_PARAMETER,
            DEFAULT_ENCODER_CONTEXT,
            SUPPORTED_ENCODERS.stream().collect(Collectors.toMap(Encoder::getName, Encoder::getMethodComponent))
        );
    }
}
