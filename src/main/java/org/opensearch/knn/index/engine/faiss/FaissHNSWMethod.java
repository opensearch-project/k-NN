/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
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
import org.opensearch.knn.index.remote.RemoteFaissHNSWIndexParameters;
import org.opensearch.knn.index.remote.RemoteIndexParameters;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.FAISS_HNSW_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * Faiss HNSW method implementation
 */
public class FaissHNSWMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(
        VectorDataType.FLOAT,
        VectorDataType.BINARY,
        VectorDataType.BYTE
    );

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
     * Get the parameters that need to be passed to the remote build service for training
     *
     * @param indexInfoParameters result of indexInfo.getParameters() to parse
     * @return Map of parameters to be used as "index_parameters"
     */
    public static RemoteIndexParameters createRemoteIndexingParameters(Map<String, Object> indexInfoParameters) {
        RemoteFaissHNSWIndexParameters.RemoteFaissHNSWIndexParametersBuilder<?, ?> builder = RemoteFaissHNSWIndexParameters.builder();
        assert (indexInfoParameters.get(SPACE_TYPE) instanceof String);
        String spaceType = (String) indexInfoParameters.get(SPACE_TYPE);
        builder.algorithm(METHOD_HNSW).spaceType(spaceType);

        Object innerParams = indexInfoParameters.get(PARAMETERS);
        assert (innerParams instanceof Map);
        Map<String, Object> innerMap = (Map<String, Object>) innerParams;
        assert (innerMap.get(METHOD_PARAMETER_EF_CONSTRUCTION) instanceof Integer);
        builder.efConstruction((Integer) innerMap.get(METHOD_PARAMETER_EF_CONSTRUCTION));
        assert (innerMap.get(METHOD_PARAMETER_EF_SEARCH) instanceof Integer);
        builder.efSearch((Integer) innerMap.get(METHOD_PARAMETER_EF_SEARCH));
        Object indexDescription = indexInfoParameters.get(INDEX_DESCRIPTION_PARAMETER);
        assert indexDescription instanceof String;
        builder.m(getMFromIndexDescription((String) indexDescription));

        return builder.build();
    }

    private static int getMFromIndexDescription(String indexDescription) {
        int commaIndex = indexDescription.indexOf(",");
        if (commaIndex == -1) {
            throw new IllegalArgumentException("Invalid index description: " + indexDescription);
        }
        String hnswPart = indexDescription.substring(0, commaIndex);
        return Integer.parseInt(hnswPart.substring(4));
    }

    /**
     * Return whether this engine/method supports remote build.
     * @param attributes Map of {@link FieldInfo} formatted attributes
     * @return true if remote build is supported, false otherwise
     * @throws IOException
     */
    static boolean supportsRemoteIndexBuild(Map<String, String> attributes) throws IOException {
        String parametersJson = attributes.get("parameters");
        String encoderName = getEncoderName(parametersJson);
        return ENCODER_FLAT.equals(encoderName);
    }

    /**
     * Gets encoder name from a {@FieldInfo parameters} map.
     * Needs to use a JSON parser since FieldInfo.attributes() is a Map of String, String.
     *
     * Example:
     * {
     *     "index_description": "HNSW12,Flat",
     *     "spaceType": "l2",
     *     "name": "hnsw",
     *     "data_type": "float",
     *     --------------------
     *     "parameters": {
     *         "ef_search": 24,
     *         "ef_construction": 28,
     *         "encoder": {
     *             "name": "flat",
     *             "parameters": {}
     *         }
     *     }
     *     --------------------
     * }
     *
     * @param parametersJson json string of parameters (inner parameter map above)
     * @return encoder name or null if not found
     * @throws IOException if the json string is not valid
     */
    private static String getEncoderName(String parametersJson) throws IOException {
        XContentParser parser = XContentType.JSON.xContent()
            .createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, parametersJson.getBytes());

        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            if (parser.currentToken() == XContentParser.Token.FIELD_NAME) {
                String fieldName = parser.currentName();

                if (PARAMETERS.equals(fieldName)) {
                    parser.nextToken();
                    if (parser.currentToken() == XContentParser.Token.START_OBJECT) {
                        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
                            if (parser.currentToken() == XContentParser.Token.FIELD_NAME) {
                                String paramName = parser.currentName();

                                if (METHOD_ENCODER_PARAMETER.equals(paramName)) {
                                    parser.nextToken();
                                    if (parser.currentToken() == XContentParser.Token.START_OBJECT) {
                                        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
                                            if (parser.currentToken() == XContentParser.Token.FIELD_NAME) {
                                                String encoderField = parser.currentName();

                                                if (NAME.equals(encoderField)) {
                                                    // .nextToken to move from the key `name` to the value.
                                                    parser.nextToken();
                                                    return parser.text();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return null;
    }
}
