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
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DocValuesType;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FP16_MAX_VALUE;
import static org.opensearch.knn.common.KNNConstants.FP16_MIN_VALUE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNValidationUtil.validateFloatVectorValue;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class KNNVectorFieldMapperUtil {

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
     * Validates and throws exception if data_type field is set in the index mapping
     * using any VectorDataType (other than float, which is default) because other
     * VectorDataTypes are only supported for lucene engine.
     *
     * @param vectorDataType VectorDataType Parameter
     */
    public static void validateVectorDataTypeWithEngine(ParametrizedFieldMapper.Parameter<VectorDataType> vectorDataType) {
        if (VectorDataType.FLOAT == vectorDataType.getValue()) {
            return;
        }
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT,
                "[%s] field with value [%s] is only supported for [%s] engine",
                VECTOR_DATA_TYPE_FIELD,
                vectorDataType.getValue().getValue(),
                LUCENE_NAME
            )
        );
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
                    "[%s] field with value [%s] is only supported for [%s] engine",
                    VECTOR_DATA_TYPE_FIELD,
                    vectorDataType.getValue().getValue(),
                    LUCENE_NAME
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

    public static void addStoredFieldForVectorField(ParseContext context, FieldType fieldType, String mapperName, Field vectorField) {
        if (fieldType.stored()) {
            context.doc().add(new StoredField(mapperName, vectorField.toString()));
        }
    }
}
