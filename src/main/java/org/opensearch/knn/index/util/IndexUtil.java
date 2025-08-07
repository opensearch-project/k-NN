/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;

import org.apache.commons.lang.StringUtils;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.ValidationException;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.request.MethodParameter;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;
import org.opensearch.knn.jni.JNIService;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ADC_ENABLED_FAISS_INDEX_INTERNAL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.BYTES_PER_KILOBYTES;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.common.KNNConstants.HNSW_ALGO_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.QUANTIZATION_LEVEL_FAISS_INDEX_LOAD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE_FAISS_INDEX_LOAD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_PARAMETER;

public class IndexUtil {

    public static final String MODEL_NODE_ASSIGNMENT_KEY = KNNConstants.MODEL_NODE_ASSIGNMENT;
    public static final String MODEL_METHOD_COMPONENT_CONTEXT_KEY = KNNConstants.MODEL_METHOD_COMPONENT_CONTEXT;

    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_IGNORE_UNMAPPED = Version.V_2_11_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_MODEL_NODE_ASSIGNMENT = Version.V_2_12_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_MODEL_METHOD_COMPONENT_CONTEXT = Version.V_2_13_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_RADIAL_SEARCH = Version.V_2_14_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_METHOD_PARAMETERS = Version.V_2_16_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_MODEL_VECTOR_DATA_TYPE = Version.V_2_16_0;
    private static final Version MINIMAL_RESCORE_FEATURE = Version.V_2_17_0;
    private static final Version MINIMAL_MODE_AND_COMPRESSION_FEATURE = Version.V_2_17_0;
    private static final Version MINIMAL_TOP_LEVEL_SPACE_TYPE_FEATURE = Version.V_2_17_0;
    private static final Version MINIMAL_SUPPORTED_VERSION_FOR_MODEL_VERSION = Version.V_2_17_0;
    private static final Version MINIMAL_EXPAND_NESTED_FEATURE = Version.V_2_19_0;
    private static final Version MINIMAL_TOP_LEVEL_ENGINE_FEATURE = Version.V_3_2_0;
    // public so neural search can access it
    public static final Map<String, Version> minimalRequiredVersionMap = initializeMinimalRequiredVersionMap();
    public static final Set<VectorDataType> VECTOR_DATA_TYPES_NOT_SUPPORTING_ENCODERS = Set.of(VectorDataType.BINARY, VectorDataType.BYTE);

    /**
     * Determines the size of a file on disk in kilobytes
     *
     * @param filePath path to the file
     * @return file size in kilobytes
     */
    public static int getFileSizeInKB(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            return 0;
        }
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) {
            return 0;
        }

        return Math.toIntExact((file.length() / BYTES_PER_KILOBYTES) + 1L); // Add one so that integer division rounds up
    }

    /**
     * Validate that a field is a k-NN vector field and has the expected dimension
     *
     * @param indexMetadata metadata for index to validate
     * @param field field name to validate
     * @param expectedDimension expected dimension of the field. If this value is negative, dimension will not be
     *                          checked
     * @param modelDao used to look up dimension if field uses a model for initialization. Can be null if
     *                 expectedDimension is negative
     * @return ValidationException exception produced by field validation
     */
    @SuppressWarnings("unchecked")
    public static ValidationException validateKnnField(
        IndexMetadata indexMetadata,
        String field,
        int expectedDimension,
        ModelDao modelDao,
        VectorDataType trainRequestVectorDataType,
        KNNMethodContext trainRequestKnnMethodContext
    ) {
        // Index metadata should not be null
        if (indexMetadata == null) {
            throw new IllegalArgumentException("IndexMetadata should not be null");
        }

        ValidationException exception = new ValidationException();

        // Check the mapping
        MappingMetadata mappingMetadata = indexMetadata.mapping();
        if (mappingMetadata == null) {
            exception.addValidationError("Invalid index. Index does not contain a mapping");
            return exception;
        }

        // The mapping output *should* look like this:
        // "{properties={field={type=knn_vector, dimension=8}}}"
        Map<String, Object> properties = (Map<String, Object>) mappingMetadata.getSourceAsMap().get("properties");

        if (properties == null) {
            exception.addValidationError("Properties in map does not exists. This is unexpected");
            return exception;
        }

        // Check field path is valid
        if (StringUtils.isEmpty(field)) {
            exception.addValidationError(String.format(Locale.ROOT, "Field path is empty."));
            return exception;
        }

        Object fieldMapping = getFieldMapping(properties, field);

        // Check field existence
        if (fieldMapping == null) {
            exception.addValidationError(String.format("Field \"%s\" does not exist.", field));
            return exception;
        }

        // Check if field is a map. If not, that is a problem
        if (!(fieldMapping instanceof Map)) {
            exception.addValidationError(String.format("Field info for \"%s\" is not a map.", field));
            return exception;
        }

        Map<String, Object> fieldMap = (Map<String, Object>) fieldMapping;

        // Check fields type is knn_vector
        Object type = fieldMap.get("type");

        if (!(type instanceof String) || !KNNVectorFieldMapper.CONTENT_TYPE.equals(type)) {
            exception.addValidationError(String.format("Field \"%s\" is not of type %s.", field, KNNVectorFieldMapper.CONTENT_TYPE));
            return exception;
        }

        if (trainRequestVectorDataType != null) {
            VectorDataType trainIndexDataType = getVectorDataTypeFromFieldMapping(fieldMap);

            if (trainIndexDataType != trainRequestVectorDataType) {
                exception.addValidationError(
                    String.format(
                        Locale.ROOT,
                        "Field \"%s\" has data type %s, which is different from data type used in the training request: %s",
                        field,
                        trainIndexDataType.getValue(),
                        trainRequestVectorDataType.getValue()
                    )
                );
                return exception;
            }

            // Block binary and byte vector data type for any encoder
            if (trainRequestKnnMethodContext != null) {
                MethodComponentContext methodComponentContext = trainRequestKnnMethodContext.getMethodComponentContext();
                Map<String, Object> parameters = methodComponentContext.getParameters();

                if (parameters != null && parameters.containsKey(KNNConstants.METHOD_ENCODER_PARAMETER)) {
                    MethodComponentContext encoder = (MethodComponentContext) parameters.get(KNNConstants.METHOD_ENCODER_PARAMETER);
                    if (encoder != null
                        && VECTOR_DATA_TYPES_NOT_SUPPORTING_ENCODERS.contains(trainRequestVectorDataType)
                        && ENCODER_FLAT.equals(encoder.getName()) == false) {
                        exception.addValidationError(
                            String.format(
                                Locale.ROOT,
                                "encoder is not supported for vector data type [%s]",
                                trainRequestVectorDataType.getValue()
                            )
                        );
                        return exception;
                    }
                }
            }
        }

        // Return if dimension does not need to be checked
        if (expectedDimension < 0) {
            return null;
        }

        // Check that the dimension of the method passed in matches that of the model
        Object dimension = fieldMap.get(KNNConstants.DIMENSION);

        // If dimension is null, the training index/field could use a model. In this case, we need to get the model id
        // for the index and then fetch its dimension from the models metadata
        if (dimension == null) {

            String modelId = (String) fieldMap.get(KNNConstants.MODEL_ID);

            if (modelId == null) {
                exception.addValidationError(String.format("Field \"%s\" does not have a dimension set.", field));
                return exception;
            }

            if (modelDao == null) {
                throw new IllegalArgumentException(String.format("Field \"%s\" uses model. modelDao cannot be null.", field));
            }

            ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
            if (!ModelUtil.isModelCreated(modelMetadata)) {
                exception.addValidationError(String.format("Model \"%s\" for field \"%s\" is not created.", modelId, field));
                return exception;
            }

            dimension = modelMetadata.getDimension();
            if ((Integer) dimension != expectedDimension) {
                exception.addValidationError(
                    String.format(
                        "Field \"%s\" has dimension %d, which is different from " + "dimension specified in the training request: %d",
                        field,
                        dimension,
                        expectedDimension
                    )
                );
                return exception;
            }

            return null;
        }

        // If the dimension was found in training fields mapping, check that it equals the models proposed dimension.
        if ((Integer) dimension != expectedDimension) {
            exception.addValidationError(
                String.format(
                    "Field \"%s\" has dimension %d, which is different from " + "dimension specified in the training request: %d",
                    field,
                    dimension,
                    expectedDimension
                )
            );
            return exception;
        }

        return null;
    }

    /**
     * Gets the load time parameters for a given engine.
     *
     * @param spaceType Space for this particular segment
     * @param knnEngine Engine used for the native library indices being loaded in
     * @param indexName Name of OpenSearch index that the segment files belong to
     * @param vectorDataType Vector data type for this particular segment
     * @return load parameters that will be passed to the JNI.
     */
    public static Map<String, Object> getParametersAtLoading(
        SpaceType spaceType,
        KNNEngine knnEngine,
        String indexName,
        VectorDataType vectorDataType,
        SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) {
        Map<String, Object> loadParameters = Maps.newHashMap(ImmutableMap.of(SPACE_TYPE, spaceType.getValue()));

        // For nmslib, we need to add the dynamic ef_search parameter that needs to be passed in when the
        // hnsw graphs are loaded into memory
        if (KNNEngine.NMSLIB.equals(knnEngine)) {
            loadParameters.put(HNSW_ALGO_EF_SEARCH, KNNSettings.getEfSearchParam(indexName));
        }
        loadParameters.put(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());

        if (SegmentLevelQuantizationUtil.isAdcEnabled(segmentLevelQuantizationInfo)) {
            loadParameters.put(ADC_ENABLED_FAISS_INDEX_INTERNAL_PARAMETER, true);
            final String quantizationLevel = segmentLevelQuantizationInfo.getQuantizationParams().getTypeIdentifier();

            loadParameters.put(QUANTIZATION_LEVEL_FAISS_INDEX_LOAD_PARAMETER, quantizationLevel);
            loadParameters.put(SPACE_TYPE_FAISS_INDEX_LOAD_PARAMETER, spaceType.getValue());
        }

        return Collections.unmodifiableMap(loadParameters);
    }

    public static boolean isClusterOnOrAfterMinRequiredVersion(String key) {
        Version minimalRequiredVersion = minimalRequiredVersionMap.get(key);
        if (minimalRequiredVersion == null) {
            return false;
        }
        return KNNClusterUtil.instance().getClusterMinVersion().onOrAfter(minimalRequiredVersion);
    }

    public static boolean isVersionOnOrAfterMinRequiredVersion(Version version, String key) {
        Version minimalRequiredVersion = minimalRequiredVersionMap.get(key);
        if (minimalRequiredVersion == null) {
            return false;
        }
        return version.onOrAfter(minimalRequiredVersion);
    }

    /**
     * Checks if index requires shared state
     *
     * @param knnEngine The knnEngine associated with the index
     * @param modelId The modelId associated with the index
     * @param indexAddr Address to check if loaded index requires shared state
     * @return true if state can be shared; false otherwise
     */
    public static boolean isSharedIndexStateRequired(KNNEngine knnEngine, String modelId, long indexAddr) {
        if (StringUtils.isEmpty(modelId)) {
            return false;
        }
        return JNIService.isSharedIndexStateRequired(indexAddr, knnEngine);
    }

    /**
     * Tell if it is binary index or not
     *
     * @param knnEngine knn engine associated with an index
     * @param parameters parameters associated with an index
     * @return true if it is binary index
     */
    public static boolean isBinaryIndex(KNNEngine knnEngine, Map<String, Object> parameters) {
        return KNNEngine.FAISS == knnEngine
            && parameters.get(VECTOR_DATA_TYPE_FIELD) != null
            && parameters.get(VECTOR_DATA_TYPE_FIELD).toString().equals(VectorDataType.BINARY.getValue());
    }

    /**
     * Return whether Asymmetric Distance Computation (ADC) is enabled for this index.
     * @param knnEngine knn engine associated with an index
     * @param parameters parameters associated with an index
     * @return true if ADC is enabled
     */
    public static boolean isADCEnabled(KNNEngine knnEngine, Map<String, Object> parameters) {
        return KNNEngine.FAISS == knnEngine
            && parameters != null
            && parameters.get(ADC_ENABLED_FAISS_INDEX_INTERNAL_PARAMETER) != null
            && (boolean) parameters.get(ADC_ENABLED_FAISS_INDEX_INTERNAL_PARAMETER);
    }

    /**
     * Update vector data type into parameters
     *
     * @param parameters parameters associated with an index
     * @param vectorDataType vector data type
     */
    public static void updateVectorDataTypeToParameters(Map<String, Object> parameters, VectorDataType vectorDataType) {
        if (VectorDataType.BINARY == vectorDataType) {
            parameters.put(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        }
        if (VectorDataType.BYTE == vectorDataType) {
            parameters.put(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        }
    }

    /**
     * This method retrieves the field mapping by a given field path from the index metadata.
     *
     * @param properties Index metadata mapping properties.
     * @param fieldPath The field path string that make up the path to the field mapping. e.g. "a.b.field" or "field".
     *                  The field path is applied and checked in OpenSearch, so it is guaranteed to be valid.
     *
     * @return           The field mapping object if found, or null if the field is not found in the index metadata.
     */
    private static Object getFieldMapping(final Map<String, Object> properties, final String fieldPath) {
        String[] fieldPaths = fieldPath.split("\\.");
        Object currentFieldMapping = properties;

        // Iterate through the field path list to retrieve the field mapping.
        for (String path : fieldPaths) {
            currentFieldMapping = ((Map<String, Object>) currentFieldMapping).get(path);
            if (currentFieldMapping == null) {
                return null;
            }

            if (currentFieldMapping instanceof Map<?, ?>) {
                Object possibleProperties = ((Map<String, Object>) currentFieldMapping).get("properties");
                if (possibleProperties instanceof Map<?, ?>) {
                    currentFieldMapping = possibleProperties;
                }
            }
        }

        return currentFieldMapping;
    }

    /**
     *  This method is used to get the vector data type from field mapping
     * @param fieldMap field mapping
     * @return vector data type
     */
    private static VectorDataType getVectorDataTypeFromFieldMapping(Map<String, Object> fieldMap) {
        if (fieldMap.containsKey(VECTOR_DATA_TYPE_FIELD)) {
            return VectorDataType.get((String) fieldMap.get(VECTOR_DATA_TYPE_FIELD));
        }
        return VectorDataType.DEFAULT;
    }

    /**
     * Initialize the minimal required version map
     *
     * @return minimal required version map
     */
    private static Map<String, Version> initializeMinimalRequiredVersionMap() {
        final Map<String, Version> versionMap = new HashMap<>() {
            {
                put("ignore_unmapped", MINIMAL_SUPPORTED_VERSION_FOR_IGNORE_UNMAPPED);
                put(MODEL_NODE_ASSIGNMENT_KEY, MINIMAL_SUPPORTED_VERSION_FOR_MODEL_NODE_ASSIGNMENT);
                put(MODEL_METHOD_COMPONENT_CONTEXT_KEY, MINIMAL_SUPPORTED_VERSION_FOR_MODEL_METHOD_COMPONENT_CONTEXT);
                put(KNNConstants.RADIAL_SEARCH_KEY, MINIMAL_SUPPORTED_VERSION_FOR_RADIAL_SEARCH);
                put(KNNConstants.METHOD_PARAMETER, MINIMAL_SUPPORTED_VERSION_FOR_METHOD_PARAMETERS);
                put(KNNConstants.MODEL_VECTOR_DATA_TYPE_KEY, MINIMAL_SUPPORTED_VERSION_FOR_MODEL_VECTOR_DATA_TYPE);
                put(RESCORE_PARAMETER, MINIMAL_RESCORE_FEATURE);
                put(KNNConstants.MINIMAL_MODE_AND_COMPRESSION_FEATURE, MINIMAL_MODE_AND_COMPRESSION_FEATURE);
                put(KNNConstants.TOP_LEVEL_SPACE_TYPE_FEATURE, MINIMAL_TOP_LEVEL_SPACE_TYPE_FEATURE);
                put(KNNConstants.MODEL_VERSION, MINIMAL_SUPPORTED_VERSION_FOR_MODEL_VERSION);
                put(EXPAND_NESTED, MINIMAL_EXPAND_NESTED_FEATURE);
                put(KNNConstants.TOP_LEVEL_ENGINE_FEATURE, MINIMAL_TOP_LEVEL_ENGINE_FEATURE);
            }
        };

        for (final MethodParameter methodParameter : MethodParameter.values()) {
            if (methodParameter.getVersion() != null) {
                versionMap.put(methodParameter.getName(), methodParameter.getVersion());
            }
        }
        return Collections.unmodifiableMap(versionMap);
    }

    /**
     * Tell if it is byte index or not
     *
     * @param parameters parameters associated with an index
     * @return true if it is binary index
     */
    public static boolean isByteIndex(Map<String, Object> parameters) {
        return parameters.getOrDefault(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .toString()
            .equals(VectorDataType.BYTE.getValue());
    }

    public static boolean isDerivedEnabledForIndex(MapperService mapperService) {
        if (mapperService == null) {
            return false;
        }

        if (mapperService.documentMapper().sourceMapper().enabled() == false) {
            return false;
        }

        if (KNNSettings.isKNNDerivedSourceEnabled(mapperService.getIndexSettings().getSettings()) == false) {
            return false;
        }

        // We do not support derived fields for seg rep with node to node replication enabled. This is because of
        // special handling of source during indexing on segrep replicas with respect to the translog.
        if (mapperService.getIndexSettings().isSegRepLocalEnabled()) {
            return false;
        }
        return true;
    }

    public static boolean isDerivedEnabledForField(KNNVectorFieldType knnVectorFieldType, MapperService mapperService) {
        // Skip copy to fields
        if (mapperService.documentMapper().mappers().getMapper(knnVectorFieldType.name()) instanceof FieldMapper mapper) {
            return mapper.copyTo() == null || mapper.copyTo().copyToFields() == null || mapper.copyTo().copyToFields().isEmpty() != false;
        }
        return true;
    }
}
