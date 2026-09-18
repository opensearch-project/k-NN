/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_BF16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.engine.faiss.FaissBF16Util.validateBF16VectorValue;

public class FaissBF16UtilTests extends KNNTestCase {

    public void testValidateBF16VectorValue_withNaN_thenThrowException() {
        expectThrows(IllegalArgumentException.class, () -> validateBF16VectorValue(Float.NaN));
    }

    public void testValidateBF16VectorValue_withPositiveInfinity_thenThrowException() {
        expectThrows(IllegalArgumentException.class, () -> validateBF16VectorValue(Float.POSITIVE_INFINITY));
    }

    public void testValidateBF16VectorValue_withNegativeInfinity_thenThrowException() {
        expectThrows(IllegalArgumentException.class, () -> validateBF16VectorValue(Float.NEGATIVE_INFINITY));
    }

    public void testValidateBF16VectorValue_withFiniteValues_thenSucceed() {
        // Since BF16 has the same exponent range as float32, all finite float values are valid
        validateBF16VectorValue(0.0f);
        validateBF16VectorValue(1.0f);
        validateBF16VectorValue(-1.0f);
        validateBF16VectorValue(65504.0f);
        validateBF16VectorValue(-65504.0f);
        validateBF16VectorValue(Float.MAX_VALUE);
        validateBF16VectorValue(-Float.MAX_VALUE);
        validateBF16VectorValue(Float.MIN_VALUE);
    }

    public void testBF16Validator_validateByte_thenThrowIllegalStateException() {
        IllegalStateException ex = expectThrows(IllegalStateException.class, () -> FaissBF16Util.BF16_VALIDATOR.validateByte(1.0f));
        assertEquals("BF16_VALIDATOR should only be used for float vectors", ex.getMessage());
    }

    public void testIsFaissSQbf16_whenEncoderIsSQBF16_thenReturnTrue() {
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_BF16));
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, params);

        assertTrue(FaissBF16Util.isFaissSQbf16(methodComponentContext));
    }

    public void testIsFaissSQbf16_whenEncoderIsSQFP16_thenReturnFalse() {
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, Map.of(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16));
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, params);

        assertFalse(FaissBF16Util.isFaissSQbf16(methodComponentContext));
    }

    public void testIsFaissSQbf16_whenNoEncoderParameter_thenReturnFalse() {
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, Collections.emptyMap());

        assertFalse(FaissBF16Util.isFaissSQbf16(methodComponentContext));
    }

    public void testIsFaissSQbf16_whenEncoderIsNotSQ_thenReturnFalse() {
        MethodComponentContext encoderContext = new MethodComponentContext("pq", Map.of(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_BF16));
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, params);

        assertFalse(FaissBF16Util.isFaissSQbf16(methodComponentContext));
    }
}
