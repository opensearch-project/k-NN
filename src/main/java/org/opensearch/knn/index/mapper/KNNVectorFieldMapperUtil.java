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

import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DocValuesType;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class KNNVectorFieldMapperUtil {
    /**
     * Validate the float vector value and throw exception if it is not a number or not in the finite range.
     *
     * @param value  float vector value
     */
    public static void validateFloatVectorValue(float value) {
        if (Float.isNaN(value)) {
            throw new IllegalArgumentException("KNN vector values cannot be NaN");
        }

        if (Float.isInfinite(value)) {
            throw new IllegalArgumentException("KNN vector values cannot be infinity");
        }
    }

    /**
     * Validate the float vector value in the byte range if it is a finite number,
     * with no decimal values and in the byte range of [-128 to 127]. If not throw IllegalArgumentException.
     *
     * @param value  float value in byte range
     */
    public static void validateByteVectorValue(float value) {
        validateFloatVectorValue(value);
        if (value % 1 != 0) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.BYTE.getValue()
                )

            );
        }
        if ((int) value < Byte.MIN_VALUE || (int) value > Byte.MAX_VALUE) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [%d, %d]",
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.BYTE.getValue(),
                    Byte.MIN_VALUE,
                    Byte.MAX_VALUE
                )
            );
        }
    }

    /**
     * Validate if the given vector size matches with the dimension provided in mapping.
     *
     * @param dimension dimension of vector
     * @param vectorSize size of the vector
     */
    public static void validateVectorDimension(int dimension, int vectorSize) {
        if (dimension != vectorSize) {
            String errorMessage = String.format(Locale.ROOT, "Vector dimension mismatch. Expected: %d, Given: %d", dimension, vectorSize);
            throw new IllegalArgumentException(errorMessage);
        }

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

    public static void addStoredFieldForVectorField(
        ParseContext context,
        FieldType fieldType,
        String mapperName,
        String vectorFieldAsString
    ) {
        if (fieldType.stored()) {
            context.doc().add(new StoredField(mapperName, vectorFieldAsString));
        }
    }
}
