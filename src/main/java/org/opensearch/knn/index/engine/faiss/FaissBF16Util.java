/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_BF16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNValidationUtil.validateFloatVectorValue;

public class FaissBF16Util {

    // Supress test coverage warning
    private FaissBF16Util() {}

    // Validates if it is a finite number. BF16 has the same exponent range as float32,
    static PerDimensionValidator BF16_VALIDATOR = new PerDimensionValidator() {
        @Override
        public void validate(float value) {
            validateBF16VectorValue(value);
        }

        @Override
        public void validateByte(float value) {
            throw new IllegalStateException("BF16_VALIDATOR should only be used for float vectors");
        }
    };

    // If the encoder parameter, "clip" is set to True, if the vector value is not finite then it will be
    // rejected. Since BF16 has the same range as float32, clipping is a no-op for finite values.
    static PerDimensionProcessor CLIP_TO_BF16_PROCESSOR = new PerDimensionProcessor() {
        @Override
        public float process(float value) {
            return clipVectorValueToBF16Range(value);
        }

        @Override
        public float processByte(float value) {
            throw new IllegalStateException("CLIP_TO_BF16_PROCESSOR should not be called with byte type");
        }
    };

    /**
     * Validate the float vector value. Since BF16 has the same exponent range as float32,
     * any finite float value is valid. This just checks for NaN/Infinity.
     *
     * @param value  float vector value
     * @return  vector value (BF16 range is same as float32 for finite values)
     */
    public static float clipVectorValueToBF16Range(float value) {
        validateFloatVectorValue(value);
        return value;
    }

    /**
     * Validate the float vector value and throw exception if it is not a number or not in the finite range.
     * Since BF16 has the same exponent range as float32, all finite float32 values are representable.
     *
     * @param value float vector value
     */
    public static void validateBF16VectorValue(float value) {
        validateFloatVectorValue(value);
    }

    /**
     * Verify mapping and return true if it is a "faiss" Index using "sq" encoder of type "bf16"
     *
     * @param methodComponentContext MethodComponentContext
     * @return true if it is a "faiss" Index using "sq" encoder of type "bf16"
     */
    static boolean isFaissSQbf16(MethodComponentContext methodComponentContext) {
        MethodComponentContext encoderContext = FaissFP16Util.extractEncoderMethodComponentContext(methodComponentContext);
        if (encoderContext == null) {
            return false;
        }

        // returns true if encoder name is "sq" and type is "bf16"
        return ENCODER_SQ.equals(encoderContext.getName())
            && FAISS_SQ_ENCODER_BF16.equals(encoderContext.getParameters().getOrDefault(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16));
    }

    /**
     * Verify mapping and return the value of "clip" parameter(default false) for a "faiss" Index
     * using "sq" encoder of type "bf16".
     *
     * @param methodComponentContext MethodComponentContext
     * @return boolean value of "clip" parameter
     */
    static boolean isFaissSQClipToBF16RangeEnabled(MethodComponentContext methodComponentContext) {
        MethodComponentContext encoderContext = FaissFP16Util.extractEncoderMethodComponentContext(methodComponentContext);
        if (encoderContext == null) {
            return false;
        }
        return (boolean) encoderContext.getParameters().getOrDefault(FAISS_SQ_CLIP, false);
    }
}
