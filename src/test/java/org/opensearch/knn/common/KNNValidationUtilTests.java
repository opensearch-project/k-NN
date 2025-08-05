/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;

import static org.hamcrest.Matchers.containsString;

public class KNNValidationUtilTests extends KNNTestCase {
    public void testValidateVectorDimension_whenBinary_thenVectorSizeShouldBeEightTimesLarger() {
        int vectorLength = randomInt(100) + 1;
        Exception ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNValidationUtil.validateVectorDimension(vectorLength, vectorLength, VectorDataType.BINARY)
        );
        assertThat(
            ex.getMessage(),
            containsString("The dimension of the binary vector must be 8 times the length of the provided vector.")
        );

        // Expect no exception
        KNNValidationUtil.validateVectorDimension(vectorLength * Byte.SIZE, vectorLength, VectorDataType.BINARY);
    }

    public void testValidateVectorDimension_whenNonBinary_thenVectorSizeShouldBeSameAsDimension() {
        int dimension = randomInt(100);
        VectorDataType vectorDataType = randomInt(1) == 0 ? VectorDataType.FLOAT : VectorDataType.BYTE;
        Exception ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNValidationUtil.validateVectorDimension(dimension, dimension + 1, vectorDataType)
        );
        assertThat(ex.getMessage(), containsString("Vector dimension mismatch"));

        // Expect no exception
        KNNValidationUtil.validateVectorDimension(dimension, dimension, vectorDataType);
    }

    public void testValidateHalfFloatVectorValue_whenValidValue_thenNoException() {
        // Test valid values within FP16 range
        float randomValue = -65504.0f + (float) Math.random() * (65504.0f - (-65504.0f));

        KNNValidationUtil.validateHalfFloatVectorValue(0.0f);
        KNNValidationUtil.validateHalfFloatVectorValue(1.0f);
        KNNValidationUtil.validateHalfFloatVectorValue(-1.0f);
        KNNValidationUtil.validateHalfFloatVectorValue(65504.0f); // FP16_MAX_VALUE
        KNNValidationUtil.validateHalfFloatVectorValue(-65504.0f); // FP16_MIN_VALUE
        KNNValidationUtil.validateHalfFloatVectorValue(randomValue);
    }

    public void testValidateHalfFloatVectorValue_whenValueTooLarge_thenThrowException() {
        Exception ex = expectThrows(IllegalArgumentException.class, () -> KNNValidationUtil.validateHalfFloatVectorValue(65505.0f));
        assertThat(ex.getMessage(), containsString("KNN vector values are not within in the half float range"));
        assertThat(ex.getMessage(), containsString("65504"));
    }

    public void testValidateHalfFloatVectorValue_whenValueTooSmall_thenThrowException() {
        Exception ex = expectThrows(IllegalArgumentException.class, () -> KNNValidationUtil.validateHalfFloatVectorValue(-65505.0f));
        assertThat(ex.getMessage(), containsString("KNN vector values are not within in the half float range"));
        assertThat(ex.getMessage(), containsString("-65504"));
    }

    public void testValidateHalfFloatVectorValue_whenNaN_thenThrowException() {
        Exception ex = expectThrows(IllegalArgumentException.class, () -> KNNValidationUtil.validateHalfFloatVectorValue(Float.NaN));
        assertThat(ex.getMessage(), containsString("cannot be NaN"));
    }

    public void testValidateHalfFloatVectorValue_whenPositiveInfinity_thenThrowException() {
        Exception ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNValidationUtil.validateHalfFloatVectorValue(Float.POSITIVE_INFINITY)
        );
        assertThat(ex.getMessage(), containsString("cannot be infinity"));
    }

    public void testValidateHalfFloatVectorValue_whenNegativeInfinity_thenThrowException() {
        Exception ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNValidationUtil.validateHalfFloatVectorValue(Float.NEGATIVE_INFINITY)
        );
        assertThat(ex.getMessage(), containsString("cannot be infinity"));
    }
}
