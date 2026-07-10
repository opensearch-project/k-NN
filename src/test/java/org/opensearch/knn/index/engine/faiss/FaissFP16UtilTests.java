/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.Encoder.QuantizationBits;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.FP16_MAX_VALUE;
import static org.opensearch.knn.common.KNNConstants.FP16_MIN_VALUE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;
import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.clipVectorValueToFP16Range;
import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.isFaissSQfp16;
import static org.opensearch.knn.index.engine.faiss.FaissFP16Util.validateFP16VectorValue;

public class FaissFP16UtilTests extends KNNTestCase {

    public void testValidateFp16VectorValue_outOfRange_throwsException() {
        IllegalArgumentException ex = expectThrows(IllegalArgumentException.class, () -> validateFP16VectorValue(65505.25f));
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );

        IllegalArgumentException ex1 = expectThrows(IllegalArgumentException.class, () -> validateFP16VectorValue(-65525.65f));
        assertTrue(
            ex1.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "encoder name is set as [%s] and type is set as [%s] in index mapping. But, KNN vector values are not within in the FP16 range [%f, %f]",
                        ENCODER_SQ,
                        FAISS_SQ_ENCODER_FP16,
                        FP16_MIN_VALUE,
                        FP16_MAX_VALUE
                    )
                )
        );
    }

    public void testClipVectorValuetoFP16Range_succeed() {
        assertEquals(65504.0f, clipVectorValueToFP16Range(65504.10f), 0.0f);
        assertEquals(65504.0f, clipVectorValueToFP16Range(1000000.89f), 0.0f);
        assertEquals(-65504.0f, clipVectorValueToFP16Range(-65504.10f), 0.0f);
        assertEquals(-65504.0f, clipVectorValueToFP16Range(-1000000.89f), 0.0f);
    }

    public void testIsFaissSQfp16_nullContext_returnsFalse() {
        assertFalse(isFaissSQfp16(null));
    }

    public void testIsFaissSQfp16_noEncoder_returnsFalse() {
        MethodComponentContext methodContext = new MethodComponentContext(METHOD_HNSW, new HashMap<>());
        assertFalse(isFaissSQfp16(methodContext));
    }

    public void testIsFaissSQfp16_nonSQEncoder_returnsFalse() {
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_FLAT, new HashMap<>());
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        MethodComponentContext methodContext = new MethodComponentContext(METHOD_HNSW, params);
        assertFalse(isFaissSQfp16(methodContext));
    }

    public void testIsFaissSQfp16_sqEncoderWithBits1_returnsFalse() {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(SQ_BITS, QuantizationBits.ONE.getValue());
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        MethodComponentContext methodContext = new MethodComponentContext(METHOD_HNSW, params);
        assertFalse(isFaissSQfp16(methodContext));
    }

    public void testIsFaissSQfp16_sqEncoderWithTypeFp16_returnsTrue() {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16);
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        MethodComponentContext methodContext = new MethodComponentContext(METHOD_HNSW, params);
        assertTrue(isFaissSQfp16(methodContext));
    }

    public void testIsFaissSQfp16_sqEncoderWithBits16_returnsTrue() {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(SQ_BITS, QuantizationBits.SIXTEEN.getValue());
        encoderParams.put(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16);
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_SQ, encoderParams);
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        MethodComponentContext methodContext = new MethodComponentContext(METHOD_HNSW, params);
        assertTrue(isFaissSQfp16(methodContext));
    }

}
