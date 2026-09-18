/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import lombok.extern.slf4j.Slf4j;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultHnswSearchContext;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
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
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_HNSW_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Faiss HNSW method implementation
 */
@Slf4j
public class FaissHNSWMethod extends AbstractFaissMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(
        VectorDataType.FLOAT,
        VectorDataType.BINARY,
        VectorDataType.BYTE,
        VectorDataType.HALF_FLOAT
    );

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.HAMMING,
        SpaceType.L2,
        SpaceType.INNER_PRODUCT,
        SpaceType.COSINESIMIL
    );

    private final static MethodComponentContext DEFAULT_ENCODER_CONTEXT = new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap());

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

    @Override
    public ValidationException validate(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
        if (knnMethodConfigContext.getVectorDataType() == VectorDataType.HALF_FLOAT && resolvesToSqFp16(knnMethodContext)) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                "half_float is not supported with fp16 quantization (encoder: sq, bits: 16, or no bits specified) for Faiss HNSW. "
                    + "half_float does not accept an encoder at all; use \"compression_level\": \"16x\" for SQ 1-bit, "
                    + "or \"1x\" for unquantized fp16 storage, instead."
            );
            return validationException;
        }
        return super.validate(knnMethodContext, knnMethodConfigContext);
    }

    private boolean resolvesToSqFp16(KNNMethodContext knnMethodContext) {
        MethodComponentContext encoderContext = getEncoderComponentContext(knnMethodContext);
        if (encoderContext == null || !ENCODER_SQ.equals(encoderContext.getName())) {
            return false;
        }
        Object bitsObj = encoderContext.getParameters().get(SQ_BITS);
        return bitsObj == null || (bitsObj instanceof Integer && (Integer) bitsObj == 16);
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
    static boolean supportsRemoteIndexBuild(final Map<String, Object> parameters) {
        try {
            final VectorDataType vectorDataType = extractVectorDataType(parameters);
            final Map<String, Object> encoderMap = extractEncoderMap(parameters);

            // TODO: turn this on once half_float is supported for remote index build. Each of the
            // checks below already requires FLOAT/BINARY/BYTE, so this is stating the existing
            // behavior rather than changing it.
            if (vectorDataType == VectorDataType.HALF_FLOAT) {
                return false;
            }

            if (isSQOneBitIndex(vectorDataType, parameters)) {
                return true;
            }

            if (isFloat32Index(vectorDataType, encoderMap)) {
                return true;
            }

            if (isFloat16Index(vectorDataType, parameters)) {
                return true;
            }

            if (isBinaryIndex(vectorDataType, encoderMap)) {
                return true;
            }

            if (isQuantizedIndex(vectorDataType, encoderMap)) {
                return true;
            }

            return isByteIndex(vectorDataType, encoderMap);
        } catch (final Exception e) {
            // We don't need to rethrow this, as technically, it is not error even we hit an exception here.
            // It merely tells us that configured parameters are not set in a way that we expect for supported types.
            log.warn(e.getMessage());
        }

        return false;
    }

    private static boolean isFloat32Index(final VectorDataType vectorDataType, final Map<String, Object> encoderMap) {
        try {
            // Check whether if float32 vector data
            if (vectorDataType != VectorDataType.FLOAT) {
                return false;
            }

            // Check encoding is 'flat'
            final String encoder = getStringFromMap(encoderMap, NAME);
            return encoder.equals(ENCODER_FLAT);
        } catch (final Exception e) {
            log.debug(e.getMessage());
            // Ignore
            return false;
        }
    }

    /**
     * From indexing library parameter, it determines whether configured index is FP16, scalar quantized.
     *
     * @param parameters KNN library indexing parameters.
     * @return Trye if FP16, otherwise False.
     */
    public static boolean isFloat16Index(final VectorDataType vectorDataType, final Map<String, Object> parameters) {
        try {
            // Check whether if vector type is float
            if (vectorDataType != VectorDataType.FLOAT) {
                return false;
            }

            // Check encoding is 'sq' meaning fp32 is being scalar quantized to fp16
            final Map<String, Object> encoderMap = extractEncoderMap(parameters);
            final String encoder = getStringFromMap(encoderMap, NAME);
            if (encoder.equals(ENCODER_SQ) == false) {
                return false;
            }
            // bits is null for legacy pre-3.6.0 indexes which default to fp16
            Object bits = encoderMap.get(SQ_BITS);
            return bits == null || (bits instanceof Integer && (Integer) bits == FaissSQEncoder.Bits.SIXTEEN.getValue());
        } catch (final Exception e) {
            log.debug(e.getMessage());
            // Ignore
            return false;
        }
    }

    private static boolean isBinaryIndex(final VectorDataType vectorDataType, final Map<String, Object> encoderMap) {
        try {
            // This index type is a binary case where user ingested binary vectors (e.g. bit stream)
            // Therefore, we didn't do any quantization from our end, it is already done from user side.
            // Check whether if vector type is binary
            return vectorDataType == VectorDataType.BINARY && getStringFromMap(encoderMap, NAME).equals(ENCODER_FLAT);
        } catch (final Exception e) {
            log.warn(e.getMessage());
            // Ignore
            return false;
        }
    }

    private static boolean isQuantizedIndex(final VectorDataType vectorDataType, final Map<String, Object> encoderMap) {
        try {
            // Check whether if vector type is FLOAT
            // It is a little bit counter-intuitive, but for quantization, we set 'float' as a vector data type by the time
            // this method is called.
            if (vectorDataType != VectorDataType.FLOAT) {
                return false;
            }

            // Check encoding is empty. For the quantization case, we don't save encoder.
            return encoderMap.isEmpty();
        } catch (final Exception e) {
            log.debug(e.getMessage());
            // Ignore
            return false;
        }
    }

    private static boolean isByteIndex(final VectorDataType vectorDataType, final Map<String, Object> encoderMap) {
        try {
            // Check whether if byte index
            if (vectorDataType != VectorDataType.BYTE) {
                return false;
            }

            // Check encoding is 'flat'
            final String encoder = getStringFromMap(encoderMap, NAME);
            return encoder.equals(ENCODER_FLAT);
        } catch (final Exception e) {
            log.debug(e.getMessage());
            // Ignore
            return false;
        }
    }

    /**
     * Checks whether the given parameters represent an SQ 1-bit index (encoder: sq, bits: 1).
     *
     * TODO: Consolidate the logic in this function with {@link FaissSQEncoder#isSQOneBit} into one function,
     * so that there is a single source of truth. Currently, this is not possible because
     * {@link FaissSQEncoder#isSQOneBit} assumes the encoder object is a {@link MethodComponentContext}.
     *
     * @param vectorDataType The data type for the vector field
     * @param parameters KNN library indexing parameters
     * @return true if SQ 1 bit, false otherwise
     */
    public static boolean isSQOneBitIndex(final VectorDataType vectorDataType, final Map<String, Object> parameters) {
        try {
            if (vectorDataType != VectorDataType.FLOAT) {
                return false;
            }
            final Map<String, Object> encoderMap = extractEncoderMap(parameters);
            final String encoder = getStringFromMap(encoderMap, NAME);
            if (encoder.equals(ENCODER_SQ) == false) {
                return false;
            }
            Object bits = encoderMap.get(SQ_BITS);
            return bits instanceof Integer && (Integer) bits == FaissSQEncoder.Bits.ONE.getValue();
        } catch (final Exception e) {
            log.error("Failed to check if method parameters contain an sq encoder with bits=1", e);
            // Ignore
            return false;
        }
    }

    /**
     * Extract {@link VectorDataType} from parameter.
     *
     * @param parameters
     * @return
     */
    private static VectorDataType extractVectorDataType(final Map<String, Object> parameters) {
        // Check whether if byte index
        final String dataType = getStringFromMap(parameters, VECTOR_DATA_TYPE_FIELD);
        final VectorDataType vectorDataType = VectorDataType.get(dataType);
        return vectorDataType;
    }

    /**
     * Extract encoder map from the given parameters.
     * Ex: {
     *    ...
     *    "parameters: {
     *        "encoder": {
     *            <This blob will be returned>
     *        }
     *    }
     * }
     *
     * @param parameters
     * @return
     */
    private static Map<String, Object> extractEncoderMap(final Map<String, Object> parameters) {
        final Map<String, Object> innerMap = (Map<String, Object>) parameters.get(PARAMETERS);
        final Map<String, Object> encoderMap = (Map<String, Object>) innerMap.get(METHOD_ENCODER_PARAMETER);
        return encoderMap;
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
