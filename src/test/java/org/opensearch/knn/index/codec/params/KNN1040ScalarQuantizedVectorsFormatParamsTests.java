/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import junit.framework.TestCase;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.opensearch.knn.index.engine.MethodComponentContext;

import static org.opensearch.knn.common.KNNConstants.ENCODER_OPTIMIZED_SCALAR_QUANTIZER;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.OPTIMIZED_SCALAR_QUANTIZER_BITS;
import static org.opensearch.knn.common.KNNConstants.OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS;

import java.util.HashMap;
import java.util.Map;

public class KNN1040ScalarQuantizedVectorsFormatParamsTests extends TestCase {
    private static final int DEFAULT_MAX_CONNECTIONS = 16;
    private static final int DEFAULT_BEAM_WIDTH = 100;

    public void testInitParams_whenCalled_thenReturnDefaultParams() {
        KNN1040ScalarQuantizedVectorsFormatParams knn1040ScalarQuantizedVectorsFormatParams = new KNN1040ScalarQuantizedVectorsFormatParams(
            getParamsForConstructor(
                OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS,
                ENCODER_OPTIMIZED_SCALAR_QUANTIZER,
                OPTIMIZED_SCALAR_QUANTIZER_BITS
            ),
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        assertEquals(DEFAULT_MAX_CONNECTIONS, knn1040ScalarQuantizedVectorsFormatParams.getMaxConnections());
        assertEquals(DEFAULT_BEAM_WIDTH, knn1040ScalarQuantizedVectorsFormatParams.getBeamWidth());
        assertEquals(ENCODER_OPTIMIZED_SCALAR_QUANTIZER, knn1040ScalarQuantizedVectorsFormatParams.getEncoderName());
        assertEquals(
            ScalarEncoding.fromNumBits(OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS),
            knn1040ScalarQuantizedVectorsFormatParams.getBitEncoding()
        );
    }

    public void testValidate_returnsTrue_withDefaultOptimizedSQParams() {
        Map<String, Object> params = getParamsForConstructor(
            OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS,
            ENCODER_OPTIMIZED_SCALAR_QUANTIZER,
            OPTIMIZED_SCALAR_QUANTIZER_BITS
        );
        KNN1040ScalarQuantizedVectorsFormatParams knn1040ScalarQuantizedVectorsFormatParams = new KNN1040ScalarQuantizedVectorsFormatParams(
            params,
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        assertTrue(knn1040ScalarQuantizedVectorsFormatParams.validate(params));
    }

    public void testValidate_returnsFalse_whenNullEncoderProvided() {
        Map<String, Object> params = getParamsForConstructor(
            OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS,
            null,
            OPTIMIZED_SCALAR_QUANTIZER_BITS
        );
        KNN1040ScalarQuantizedVectorsFormatParams knn1040ScalarQuantizedVectorsFormatParams = new KNN1040ScalarQuantizedVectorsFormatParams(
            params,
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        assertFalse(knn1040ScalarQuantizedVectorsFormatParams.validate(params));
    }

    public void testValidate_returnsFalse_whenInvalidEncoderProvided() {
        Map<String, Object> params = getParamsForConstructor(
            OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS,
            "FAKE_ENCODER",
            OPTIMIZED_SCALAR_QUANTIZER_BITS
        );
        KNN1040ScalarQuantizedVectorsFormatParams knn1040ScalarQuantizedVectorsFormatParams = new KNN1040ScalarQuantizedVectorsFormatParams(
            params,
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        assertFalse(knn1040ScalarQuantizedVectorsFormatParams.validate(params));
    }

    private Map<String, Object> getParamsForConstructor(int bits, String encoder, String encoderBitString) {
        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(encoderBitString, bits);
        MethodComponentContext encoderComponentContext = new MethodComponentContext(encoder, encoderParams);
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
        return params;
    }
}
