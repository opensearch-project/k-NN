/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import lombok.extern.slf4j.Slf4j;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultHnswSearchContext;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.remoteindexbuild.model.RemoteFaissHNSWIndexParameters;
import org.opensearch.remoteindexbuild.model.RemoteIndexParameters;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.FAISS_HNSW_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Faiss HNSW method implementation
 */
@Slf4j
public class FaissHNSWMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(
        VectorDataType.FLOAT,
        VectorDataType.HALF_FLOAT,
        VectorDataType.BINARY,
        VectorDataType.BYTE
    );

    private static final Set<VectorDataType> SUPPORTED_REMOTE_INDEX_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.HAMMING,
        SpaceType.L2,
        SpaceType.INNER_PRODUCT,
        SpaceType.COSINESIMIL
    );

    private final static MethodComponentContext DEFAULT_ENCODER_CONTEXT = new MethodComponentContext(
        KNNConstants.ENCODER_FLAT,
        Collections.emptyMap()
    );

    // Package private so that the method resolving logic can access the methods
    final static Encoder FLAT_ENCODER = new FaissFlatEncoder();
    final static Encoder SQ_ENCODER = new FaissSQEncoder();
    final static Encoder HNSW_PQ_ENCODER = new FaissHNSWPQEncoder();
    final static Encoder QFRAME_BIT_ENCODER = new QFrameBitEncoder();
    final static Map<String, Encoder> SUPPORTED_ENCODERS = Map.of(
        FLAT_ENCODER.getName(),
        FLAT_ENCODER,
        SQ_ENCODER.getName(),
        SQ_ENCODER,
        HNSW_PQ_ENCODER.getName(),
        HNSW_PQ_ENCODER,
        QFRAME_BIT_ENCODER.getName(),
        QFRAME_BIT_ENCODER
    );
    final static MethodComponent HNSW_COMPONENT = initMethodComponent();

    /**
     * Constructor for FaissHNSWMethod
     *
     * @see AbstractKNNMethod
     */
    public FaissHNSWMethod() {
        super(HNSW_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new DefaultHnswSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_HNSW)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_M,
                new Parameter.IntegerParameter(METHOD_PARAMETER_M, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, (v, context) -> v > 0)
            )
            .addParameter(
                METHOD_PARAMETER_EF_CONSTRUCTION,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                METHOD_PARAMETER_EF_SEARCH,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_EF_SEARCH,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
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
        return new Parameter.MethodComponentContextParameter(
            METHOD_ENCODER_PARAMETER,
            DEFAULT_ENCODER_CONTEXT,
            SUPPORTED_ENCODERS.values().stream().collect(Collectors.toMap(Encoder::getName, Encoder::getMethodComponent))
        );
    }

    @Override
    protected Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> doGetTrainingConfigValidationSetup() {
        return (trainingConfigValidationInput) -> {

            KNNMethodContext knnMethodContext = trainingConfigValidationInput.getKnnMethodContext();
            TrainingConfigValidationOutput.TrainingConfigValidationOutputBuilder builder = TrainingConfigValidationOutput.builder();

            if (isEncoderSpecified(knnMethodContext) == false) {
                return builder.build();
            }
            Encoder encoder = SUPPORTED_ENCODERS.get(getEncoderName(knnMethodContext));
            if (encoder == null) {
                return builder.build();
            }

            return encoder.validateEncoderConfig(trainingConfigValidationInput);
        };
    }

    /**
     * Get the parameters that need to be passed to the remote build service for training from a KNNLibraryIndexingContext LibraryParameters map
     * See example map in {@link FaissHNSWMethod#supportsRemoteIndexBuild}
     * @param parameters map to parse
     * @return Map of parameters to be used as "index_parameters" in the remote build request
     */
    @SuppressWarnings("unchecked")
    public RemoteIndexParameters createRemoteIndexingParameters(Map<String, Object> parameters) {
        RemoteFaissHNSWIndexParameters.RemoteFaissHNSWIndexParametersBuilder<?, ?> builder = RemoteFaissHNSWIndexParameters.builder();
        builder.algorithm(METHOD_HNSW);
        builder.spaceType(getStringFromMap(parameters, SPACE_TYPE));

        Map<String, Object> innerParameters = (Map<String, Object>) parameters.get(PARAMETERS);
        builder.efConstruction(getIntegerFromMap(innerParameters, METHOD_PARAMETER_EF_CONSTRUCTION));
        builder.efSearch(getIntegerFromMap(innerParameters, METHOD_PARAMETER_EF_SEARCH));
        builder.m(getIntegerFromMap(innerParameters, METHOD_PARAMETER_M));
        return builder.build();
    }

    /**
     * @param parameters Map of method parameters including encoder information
     * Example JSON structure:
     * {
     *   "index_description": "HNSW12,Flat",
     *   "spaceType": "innerproduct",
     *   "name": "hnsw",
     *   "data_type": "float",
     *   "parameters": {
     *     "ef_search": 24,
     *     "ef_construction": 28,
     *     "m": 12,
     *     "encoder": {
     *       "name": "flat",
     *       "parameters": {}
     *     }
     *   }
     * }
     * @return true if the method parameters + vector data type combination is supported for remote index build
     */
    @SuppressWarnings("unchecked")
    static boolean supportsRemoteIndexBuild(Map<String, Object> parameters) {
        try {
            Map<String, Object> innerMap = (Map<String, Object>) parameters.get(PARAMETERS);
            Map<String, Object> encoderMap = (Map<String, Object>) innerMap.get(METHOD_ENCODER_PARAMETER);
            String dataType = getStringFromMap(parameters, VECTOR_DATA_TYPE_FIELD);
            String encoder = getStringFromMap(encoderMap, NAME);
            return SUPPORTED_REMOTE_INDEX_DATA_TYPES.contains(VectorDataType.get(dataType)) && ENCODER_FLAT.equals(encoder);
        } catch (IllegalArgumentException e) {
            log.error("Unrecognized indexing parameters in KNNLibraryIndexingContext", e);
            return false;
        }
    }

    /**
     * Safely retrieve an Integer from {@code map} using {@code key}
     */
    private static Integer getIntegerFromMap(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (value instanceof Integer) {
            return (Integer) value;
        }
        if (value instanceof String) {
            return Integer.parseInt((String) value);
        }
        throw new IllegalArgumentException("Could not parse value for key: " + key + " and map: " + map);
    }

    /**
     * Safely retrieve a String from {@code map} using {@code key}
     */
    private static String getStringFromMap(Map<String, Object> map, String key) throws IllegalArgumentException {
        Object value = map.get(key);
        if (value instanceof String) {
            return (String) value;
        }
        throw new IllegalArgumentException("Could not parse value for key: " + key + " and map: " + map);
    }
}
