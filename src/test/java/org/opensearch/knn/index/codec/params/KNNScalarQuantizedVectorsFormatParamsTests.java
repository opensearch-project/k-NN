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

package org.opensearch.knn.index.codec.params;

import junit.framework.TestCase;
import org.opensearch.knn.index.MethodComponentContext;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.MINIMUM_CONFIDENCE_INTERVAL;

public class KNNScalarQuantizedVectorsFormatParamsTests extends TestCase {
    private static final int DEFAULT_MAX_CONNECTIONS = 16;
    private static final int DEFAULT_BEAM_WIDTH = 100;

    public void testInitParams_whenCalled_thenReturnDefaultParams() {
        KNNScalarQuantizedVectorsFormatParams knnScalarQuantizedVectorsFormatParams = new KNNScalarQuantizedVectorsFormatParams(
            getDefaultParamsForConstructor(),
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        assertEquals(DEFAULT_MAX_CONNECTIONS, knnScalarQuantizedVectorsFormatParams.getMaxConnections());
        assertEquals(DEFAULT_BEAM_WIDTH, knnScalarQuantizedVectorsFormatParams.getBeamWidth());
        assertNull(knnScalarQuantizedVectorsFormatParams.getConfidenceInterval());
        assertTrue(knnScalarQuantizedVectorsFormatParams.isCompressFlag());
        assertEquals(LUCENE_SQ_DEFAULT_BITS, knnScalarQuantizedVectorsFormatParams.getBits());
    }

    public void testInitParams_whenCalled_thenReturnParams() {
        int m = 64;
        int efConstruction = 128;

        Map<String, Object> encoderParams = new HashMap<>();
        encoderParams.put(LUCENE_SQ_CONFIDENCE_INTERVAL, MINIMUM_CONFIDENCE_INTERVAL);
        MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_SQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
        params.put(METHOD_PARAMETER_M, m);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction);

        KNNScalarQuantizedVectorsFormatParams knnScalarQuantizedVectorsFormatParams = new KNNScalarQuantizedVectorsFormatParams(
            params,
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        assertEquals(m, knnScalarQuantizedVectorsFormatParams.getMaxConnections());
        assertEquals(efConstruction, knnScalarQuantizedVectorsFormatParams.getBeamWidth());
        assertEquals((float) MINIMUM_CONFIDENCE_INTERVAL, knnScalarQuantizedVectorsFormatParams.getConfidenceInterval());
        assertTrue(knnScalarQuantizedVectorsFormatParams.isCompressFlag());
        assertEquals(LUCENE_SQ_DEFAULT_BITS, knnScalarQuantizedVectorsFormatParams.getBits());
    }

    public void testValidate_whenCalled_thenReturnTrue() {
        Map<String, Object> params = getDefaultParamsForConstructor();
        KNNScalarQuantizedVectorsFormatParams knnScalarQuantizedVectorsFormatParams = new KNNScalarQuantizedVectorsFormatParams(
            params,
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );
        assertTrue(knnScalarQuantizedVectorsFormatParams.validate(params));
    }

    public void testValidate_whenCalled_thenReturnFalse() {
        KNNScalarQuantizedVectorsFormatParams knnScalarQuantizedVectorsFormatParams = new KNNScalarQuantizedVectorsFormatParams(
            getDefaultParamsForConstructor(),
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );
        Map<String, Object> params = new HashMap<>();

        // Return false if encoder value is null
        params.put(METHOD_ENCODER_PARAMETER, null);
        assertFalse(knnScalarQuantizedVectorsFormatParams.validate(params));

        // Return false if encoder value is not an instance of MethodComponentContext
        params.replace(METHOD_ENCODER_PARAMETER, "dummy string");
        assertFalse(knnScalarQuantizedVectorsFormatParams.validate(params));

        // Return false if encoder name is not "sq"
        MethodComponentContext encoderComponentContext = new MethodComponentContext("invalid encoder name", new HashMap<>());
        params.replace(METHOD_ENCODER_PARAMETER, encoderComponentContext);
        assertFalse(knnScalarQuantizedVectorsFormatParams.validate(params));
    }

    private Map<String, Object> getDefaultParamsForConstructor() {
        MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_SQ, new HashMap<>());
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
        return params;
    }
}
