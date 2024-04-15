/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.util.List;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.index.mapper.BinaryFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;

import java.math.BigInteger;
import java.util.Base64;

import static org.opensearch.index.mapper.NumberFieldMapper.NumberType.LONG;
import static org.opensearch.knn.common.KNNValidationUtil.validateByteVectorValue;

public class KNNScoringSpaceUtil {

    /**
     * Check if the passed in fieldType is of type NumberFieldType with numericType being Long
     *
     * @param fieldType MappedFieldType
     * @return true if fieldType is of type NumberFieldType and its numericType is Long; false otherwise
     */
    public static boolean isLongFieldType(MappedFieldType fieldType) {
        return fieldType instanceof NumberFieldMapper.NumberFieldType
            && ((NumberFieldMapper.NumberFieldType) fieldType).numericType() == LONG.numericType();
    }

    /**
     * Check if the passed in fieldType is of type BinaryFieldType
     *
     * @param fieldType MappedFieldType
     * @return true if fieldType is of type BinaryFieldType; false otherwise
     */
    public static boolean isBinaryFieldType(MappedFieldType fieldType) {
        return fieldType instanceof BinaryFieldMapper.BinaryFieldType;
    }

    /**
     * Check if the passed in fieldType is of type KNNVectorFieldType
     *
     * @param fieldType MappedFieldType
     * @return true if fieldType is of type KNNVectorFieldType; false otherwise
     */
    public static boolean isKNNVectorFieldType(MappedFieldType fieldType) {
        return fieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType;
    }

    /**
     * Convert an Object to a Long.
     *
     * @param object Object to be parsed to a Long
     * @return Long of the object
     */
    public static Long parseToLong(Object object) {
        if (object instanceof Integer) {
            return Long.valueOf((Integer) object);
        }

        if (object instanceof Long) {
            return (Long) object;
        }

        throw new IllegalArgumentException("Object cannot be parsed as a Long.");
    }

    /**
     * Convert an Object to a BigInteger.
     *
     * @param object Base64 encoded String
     * @return BigInteger containing the bytes of decoded object
     */
    public static BigInteger parseToBigInteger(Object object) {
        return new BigInteger(1, Base64.getDecoder().decode((String) object));
    }

    /**
     * Convert an Object to a float array.
     *
     * @param object Object to be converted to a float array
     * @param expectedDimensions int representing the expected dimension of this array.
     * @return float[] of the object
     */
    public static float[] parseToFloatArray(Object object, int expectedDimensions, VectorDataType vectorDataType) {
        float[] floatArray = convertVectorToPrimitive(object, vectorDataType);
        if (expectedDimensions != floatArray.length) {
            KNNCounter.SCRIPT_QUERY_ERRORS.increment();
            throw new IllegalStateException(
                "Object's dimension=" + floatArray.length + " does not match the " + "expected dimension=" + expectedDimensions + "."
            );
        }
        return floatArray;
    }

    /**
     * Converts Object vector to primitive float[]
     *
     * @param vector input vector
     * @return Float array representing the vector
     */
    @SuppressWarnings("unchecked")
    public static float[] convertVectorToPrimitive(Object vector, VectorDataType vectorDataType) {
        float[] primitiveVector = null;
        if (vector != null) {
            final List<Number> tmp = (List<Number>) vector;
            primitiveVector = new float[tmp.size()];
            for (int i = 0; i < primitiveVector.length; i++) {
                float value = tmp.get(i).floatValue();
                if (VectorDataType.BYTE == vectorDataType) {
                    validateByteVectorValue(value);
                }
                primitiveVector[i] = value;
            }
        }
        return primitiveVector;
    }

    /**
     * Calculates the magnitude of given vector
     *
     * @param inputVector input vector
     * @return Magnitude of vector
     */
    public static float getVectorMagnitudeSquared(float[] inputVector) {
        if (null == inputVector) {
            throw new IllegalStateException("vector magnitude cannot be evaluated as it is null");
        }
        float normInputVector = 0.0f;
        for (int i = 0; i < inputVector.length; i++) {
            normInputVector += inputVector[i] * inputVector[i];
        }
        return normInputVector;
    }
}
