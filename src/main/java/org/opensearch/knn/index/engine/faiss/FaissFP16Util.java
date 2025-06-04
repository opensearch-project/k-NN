/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.PerDimensionProcessor;
import org.opensearch.knn.index.mapper.PerDimensionValidator;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.FP16_MAX_VALUE;
import static org.opensearch.knn.common.KNNConstants.FP16_MIN_VALUE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
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

    /**
     * Verify mapping and return true if it is a "faiss" Index using "sq" encoder of type "fp16"
     *
     * @param methodComponentContext MethodComponentContext
     * @return true if it is a "faiss" Index using "sq" encoder of type "fp16"
     */
    static boolean isFaissSQfp16(MethodComponentContext methodComponentContext) {
        MethodComponentContext encoderContext = extractEncoderMethodComponentContext(methodComponentContext);
        if (encoderContext == null) {
            return false;
        }

        // returns true if encoder name is "sq" and type is "fp16"
        return ENCODER_SQ.equals(encoderContext.getName())
            && FAISS_SQ_ENCODER_FP16.equals(encoderContext.getParameters().getOrDefault(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16));
    }

    /**
     * Verify mapping and return the value of "clip" parameter(default false) for a "faiss" Index
     * using "sq" encoder of type "fp16".
     *
     * @param methodComponentContext MethodComponentContext
     * @return boolean value of "clip" parameter
     */
    static boolean isFaissSQClipToFP16RangeEnabled(MethodComponentContext methodComponentContext) {
        MethodComponentContext encoderContext = extractEncoderMethodComponentContext(methodComponentContext);
        if (encoderContext == null) {
            return false;
        }
        return ENCODER_SQ.equals(encoderContext.getName())
            && FAISS_SQ_ENCODER_FP16.equals(encoderContext.getParameters().getOrDefault(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16))
            && (Boolean) encoderContext.getParameters().getOrDefault(FAISS_SQ_CLIP, false);
    }

    static MethodComponentContext extractEncoderMethodComponentContext(MethodComponentContext methodComponentContext) {
        if (Objects.isNull(methodComponentContext)) {
            return null;
        }

        if (methodComponentContext.getParameters().isEmpty()) {
            return null;
        }

        Map<String, Object> methodComponentParams = methodComponentContext.getParameters();

        // The method component parameters should have an encoder
        if (!methodComponentParams.containsKey(METHOD_ENCODER_PARAMETER)) {
            return null;
        }

        // Validate if the object is of type MethodComponentContext before casting it later
        if (!(methodComponentParams.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext)) {
            return null;
        }

        return (MethodComponentContext) methodComponentParams.get(METHOD_ENCODER_PARAMETER);
    }
}
