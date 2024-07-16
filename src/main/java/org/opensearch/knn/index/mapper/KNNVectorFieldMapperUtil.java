/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.mapper;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.util.BytesRef;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import java.util.Arrays;
import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FP16_MAX_VALUE;
import static org.opensearch.knn.common.KNNConstants.FP16_MIN_VALUE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNValidationUtil.validateFloatVectorValue;

/**
 * Utility class for KNNVectorFieldMapper
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class KNNVectorFieldMapperUtil {

    private static ModelDao modelDao;

    /**
     * Initializes static instance variables
     * @param modelDao ModelDao object
     */
    public static void initialize(final ModelDao modelDao) {
        KNNVectorFieldMapperUtil.modelDao = modelDao;
    }

    /**
     * Validate the float vector value and throw exception if it is not a number or not in the finite range
     * or is not within the FP16 range of [-65504 to 65504].
     *
     * @param value float vector value
     */
    public static void validateFP16VectorValue(float value) {
        validateFloatVectorValue(value);
        if (value < FP16_MIN_VALUE || value > FP16_MAX_VALUE) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                    ENCODER_SQ,
                    FAISS_SQ_ENCODER_FP16,
                    FP16_MIN_VALUE,
                    FP16_MAX_VALUE
                )
            );
        }
    }

    /**
     * Validate the float vector value and if it is outside FP16 range,
     * then it will be clipped to FP16 range of [-65504 to 65504].
     *
     * @param value  float vector value
     * @return  vector value clipped to FP16 range
     */
    public static float clipVectorValueToFP16Range(float value) {
        validateFloatVectorValue(value);
        if (value < FP16_MIN_VALUE) return FP16_MIN_VALUE;
        if (value > FP16_MAX_VALUE) return FP16_MAX_VALUE;
        return value;
    }

    /**
     * Validates if the vector data type is supported with given method context
     *
     * @param methodContext methodContext
     * @param vectorDataType vector data type
     */
    public static void validateVectorDataType(KNNMethodContext methodContext, VectorDataType vectorDataType) {
        if (VectorDataType.FLOAT == vectorDataType) {
            return;
        }

        if (VectorDataType.BYTE == vectorDataType) {
            if (KNNEngine.LUCENE == methodContext.getKnnEngine()) {
                return;
            } else {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "[%s] field with value [%s] is only supported for [%s] engine",
                        VECTOR_DATA_TYPE_FIELD,
                        vectorDataType.getValue(),
                        LUCENE_NAME
                    )
                );
            }
        }

        if (VectorDataType.BINARY == vectorDataType) {
            if (KNNEngine.FAISS == methodContext.getKnnEngine()) {
                if (METHOD_HNSW.equals(methodContext.getMethodComponentContext().getName())) {
                    return;
                } else {
                    throw new IllegalArgumentException(
                        String.format(
                            Locale.ROOT,
                            "[%s] field with value [%s] is only supported for [%s] method",
                            VECTOR_DATA_TYPE_FIELD,
                            vectorDataType.getValue(),
                            METHOD_HNSW
                        )
                    );
                }
            } else {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "[%s] field with value [%s] is only supported for [%s] engine",
                        VECTOR_DATA_TYPE_FIELD,
                        vectorDataType.getValue(),
                        FAISS_NAME
                    )
                );
            }
        }
        throw new IllegalArgumentException("This line should not be reached");
    }

    /**
     * Validates and throws exception if index.knn is set to true in the index settings
     * using any VectorDataType (other than float, which is default) because we are using NMSLIB engine
     * for LegacyFieldMapper, and it only supports float VectorDataType
     *
     * @param knnIndexSetting index.knn setting in the index settings
     * @param vectorDataType VectorDataType Parameter
     */
    public static void validateVectorDataTypeWithKnnIndexSetting(
        boolean knnIndexSetting,
        ParametrizedFieldMapper.Parameter<VectorDataType> vectorDataType
    ) {

        if (VectorDataType.FLOAT == vectorDataType.getValue()) {
            return;
        }
        if (knnIndexSetting) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] field with value [%s] is not supported for [%s] engine",
                    VECTOR_DATA_TYPE_FIELD,
                    vectorDataType.getValue().getValue(),
                    NMSLIB_NAME
                )
            );
        }
    }

    /**
     * @param knnEngine  KNNEngine
     * @return  DocValues FieldType of type Binary
     */
    public static FieldType buildDocValuesFieldType(KNNEngine knnEngine) {
        FieldType field = new FieldType();
        field.putAttribute(KNN_ENGINE, knnEngine.getName());
        field.setDocValuesType(DocValuesType.BINARY);
        field.freeze();
        return field;
    }

    /**
     * Creates a stored field for a byte vector
     *
     * @param name field name
     * @param vector vector to be added to stored field
     */
    public static StoredField createStoredFieldForByteVector(String name, byte[] vector) {
        return new StoredField(name, vector);
    }

    /**
     * Creates a stored field for a float vector
     *
     * @param name field name
     * @param vector vector to be added to stored field
     */
    public static StoredField createStoredFieldForFloatVector(String name, float[] vector) {
        return new StoredField(name, KNNVectorSerializerFactory.getDefaultSerializer().floatToByteArray(vector));
    }

    /**
     * @param storedVector Vector representation in bytes
     * @param vectorDataType type of vector
     * @return either int[] or float[] of corresponding vector
     */
    public static Object deserializeStoredVector(BytesRef storedVector, VectorDataType vectorDataType) {
        if (VectorDataType.BYTE == vectorDataType) {
            byte[] bytes = storedVector.bytes;
            int[] byteAsIntArray = new int[bytes.length];
            Arrays.setAll(byteAsIntArray, i -> bytes[i]);
            return byteAsIntArray;
        }

        return vectorDataType.getVectorFromBytesRef(storedVector);
    }

    /**
     * Get the expected vector length from a specified knn vector field type.
     *
     * If the field is model-based, get dimensions from model metadata.
     * For binary vector, the expected vector length is dimension divided by 8
     *
     * @param knnVectorFieldType knn vector field type
     * @return expected vector length
     */
    public static int getExpectedVectorLength(final KNNVectorFieldMapper.KNNVectorFieldType knnVectorFieldType) {
        int expectedDimensions = knnVectorFieldType.getDimension();
        if (isModelBasedIndex(expectedDimensions)) {
            ModelMetadata modelMetadata = getModelMetadataForField(knnVectorFieldType);
            expectedDimensions = modelMetadata.getDimension();
        }
        return VectorDataType.BINARY == knnVectorFieldType.getVectorDataType() ? expectedDimensions / 8 : expectedDimensions;
    }

    private static boolean isModelBasedIndex(int expectedDimensions) {
        return expectedDimensions == -1;
    }

    /**
     * Returns the model metadata for a specified knn vector field
     *
     * @param knnVectorField knn vector field
     * @return the model metadata from knnVectorField
     */
    private static ModelMetadata getModelMetadataForField(final KNNVectorFieldMapper.KNNVectorFieldType knnVectorField) {
        String modelId = knnVectorField.getModelId();

        if (modelId == null) {
            throw new IllegalArgumentException(
                String.format("Field '%s' does not have model.", knnVectorField.getKnnMethodContext().getMethodComponentContext().getName())
            );
        }

        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (!ModelUtil.isModelCreated(modelMetadata)) {
            throw new IllegalArgumentException(String.format("Model ID '%s' is not created.", modelId));
        }
        return modelMetadata;
    }
}
