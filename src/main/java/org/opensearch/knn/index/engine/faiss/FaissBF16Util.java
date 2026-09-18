/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.PerDimensionValidator;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_BF16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNValidationUtil.validateFloatVectorValue;

public class FaissBF16Util {

    // Supress test coverage warning
    private FaissBF16Util() {}

    // Validates that the value is a finite number. BF16 has the same exponent range as float32,
    // so any finite float32 value is a valid BF16 value; this only rejects NaN/Infinity.
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
}
