/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.index.VectorDataType;

import static org.opensearch.knn.common.KNNValidationUtil.validateByteVectorValue;
import static org.opensearch.knn.common.KNNValidationUtil.validateHalfFloatVectorValue;
import static org.opensearch.knn.common.KNNValidationUtil.validateFloatVectorValue;

/**
 * Validates per dimension fields
 */
public interface PerDimensionValidator {
    /**
     * Validates the given float is valid for the configuration
     *
     * @param value to validate
     */
    default void validate(float value) {}

    /**
     * Validates the given float as a byte is valid for the configuration.
     *
     * @param value to validate
     */
    default void validateByte(float value) {}

    PerDimensionValidator DEFAULT_FLOAT_VALIDATOR = new PerDimensionValidator() {
        @Override
        public void validate(float value) {
            validateFloatVectorValue(value);
        }

        @Override
        public void validateByte(float value) {
            throw new IllegalStateException("DEFAULT_FLOAT_VALIDATOR should only be used for float vectors");
        }
    };

    PerDimensionValidator DEFAULT_HALF_FLOAT_VALIDATOR = new PerDimensionValidator() {
        @Override
        public void validate(float value) {
            validateHalfFloatVectorValue(value);
        }

        @Override
        public void validateByte(float value) {
            throw new IllegalStateException("DEFAULT_HALF_FLOAT_VALIDATOR should only be used for half float vectors");
        }
    };

    PerDimensionValidator DEFAULT_BYTE_VALIDATOR = new PerDimensionValidator() {
        @Override
        public void validate(float value) {
            throw new IllegalStateException("DEFAULT_BYTE_VALIDATOR should only be used for byte values");
        }

        @Override
        public void validateByte(float value) {
            validateByteVectorValue(value, VectorDataType.BYTE);
        }
    };

    PerDimensionValidator DEFAULT_BIT_VALIDATOR = new PerDimensionValidator() {
        @Override
        public void validate(float value) {
            throw new IllegalStateException("DEFAULT_BIT_VALIDATOR should only be used for byte values");
        }

        @Override
        public void validateByte(float value) {
            validateByteVectorValue(value, VectorDataType.BINARY);
        }
    };
}
