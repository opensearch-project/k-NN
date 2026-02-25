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

package org.opensearch.knn.common;

import java.util.Locale;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.VectorDataType;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class KNNValidationUtil {
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
    public static void validateByteVectorValue(float value, final VectorDataType dataType) {
        validateFloatVectorValue(value);
        if (value % 1 != 0) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                    VECTOR_DATA_TYPE_FIELD,
                    dataType.getValue()
                )

            );
        }
        if ((int) value < Byte.MIN_VALUE || (int) value > Byte.MAX_VALUE) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [%d, %d]",
                    VECTOR_DATA_TYPE_FIELD,
                    dataType.getValue(),
                    Byte.MIN_VALUE,
                    Byte.MAX_VALUE
                )
            );
        }
    }

    /**
     * Validate if the given vector size matches with the dimension provided in mapping.
     *
     * For binary index, the dimension is 8 times larger than vector size because 8 bits is packed into single byte
     *
     * @param dimension dimension of vector
     * @param vectorSize size of the vector
     * @param dataType vector data type
     */
    public static void validateVectorDimension(final int dimension, final int vectorSize, final VectorDataType dataType) {
        int actualDimension = VectorDataType.BINARY == dataType ? vectorSize * Byte.SIZE : vectorSize;
        if (dimension != actualDimension) {
            if (VectorDataType.BINARY == dataType) {
                String errorMessage = String.format(
                    Locale.ROOT,
                    "The dimension of the binary vector must be 8 times the length of the provided vector. Expected: %d, Given: %d",
                    dimension,
                    actualDimension
                );
                throw new IllegalArgumentException(errorMessage);
            } else {
                String errorMessage = String.format(
                    Locale.ROOT,
                    "Vector dimension mismatch. Expected: %d, Given: %d",
                    dimension,
                    actualDimension
                );
                throw new IllegalArgumentException(errorMessage);
            }
        }
    }
}
