/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;

import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FP16_MAX_VALUE;
import static org.opensearch.knn.common.KNNConstants.FP16_MIN_VALUE;
import static org.opensearch.knn.common.KNNValidationUtil.validateFloatVectorValue;

public class FaissFP16Util {

    // Validates if it is a finite number and within the fp16 range of [-65504 to 65504].
    static PerDimensionValidator FP16_VALIDATOR = new PerDimensionValidator() {
        @Override
        public void validate(float value) {
            validateFP16VectorValue(value);
        }

        @Override
        public void validateByte(float value) {
            throw new IllegalStateException("DEFAULT_FP16_VALIDATOR should only be used for float vectors");
        }
    };

    // If the encoder parameter, "clip" is set to True, if the vector value is outside the FP16 range then it will be
    // clipped to FP16 range.
    static PerDimensionProcessor CLIP_TO_FP16_PROCESSOR = new PerDimensionProcessor() {
        @Override
        public float process(float value) {
            return clipVectorValueToFP16Range(value);
        }

        @Override
        public float processByte(float value) {
            throw new IllegalStateException("CLIP_TO_FP16_PROCESSOR should not be called with byte type");
        }
    };

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
}
