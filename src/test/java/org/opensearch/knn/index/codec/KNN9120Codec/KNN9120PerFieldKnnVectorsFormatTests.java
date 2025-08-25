/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import junit.framework.TestCase;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.params.KNNBBQVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNScalarQuantizedVectorsFormatParams;
import org.opensearch.knn.index.codec.params.KNNVectorsFormatParams;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

public class KNN9120PerFieldKnnVectorsFormatTests extends TestCase {

    private MapperService mockMapperService;
    private KNN9120PerFieldKnnVectorsFormat format;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mockMapperService = Mockito.mock(MapperService.class);
        format = new KNN9120PerFieldKnnVectorsFormat(Optional.of(mockMapperService));
    }

    public void testConstructor_whenCalled_thenFormatCreated() {
        assertNotNull(format);
    }

    public void testVectorsFormatParams_whenCalled_thenReturnValidParams() {
        // Test regular format
        Map<String, Object> regularParams = new HashMap<>();
        regularParams.put(METHOD_PARAMETER_M, 16);
        regularParams.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        KNNVectorsFormatParams params = new KNNVectorsFormatParams(regularParams, 16, 100);
        assertEquals(16, params.getMaxConnections());
        assertEquals(100, params.getBeamWidth());

        // Test scalar quantized format
        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext sqContext = new MethodComponentContext("sq", encoderParams);

        Map<String, Object> sqParams = new HashMap<>();
        sqParams.put(METHOD_ENCODER_PARAMETER, sqContext);
        sqParams.put(METHOD_PARAMETER_M, 16);
        sqParams.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        KNNScalarQuantizedVectorsFormatParams sqFormatParams = new KNNScalarQuantizedVectorsFormatParams(sqParams, 16, 100);
        assertEquals(16, sqFormatParams.getMaxConnections());
        assertEquals(100, sqFormatParams.getBeamWidth());
    }

    public void testBBQParameterValidation_whenCalled_thenValidateCorrectly() {
        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext bbqEncoder = new MethodComponentContext(ENCODER_BBQ, encoderParams);

        // Valid BBQ parameters
        Map<String, Object> validParams = new HashMap<>();
        validParams.put(METHOD_ENCODER_PARAMETER, bbqEncoder);
        validParams.put(METHOD_PARAMETER_M, 64);
        validParams.put(METHOD_PARAMETER_EF_CONSTRUCTION, 256);

        KNNBBQVectorsFormatParams bbqParams = new KNNBBQVectorsFormatParams(validParams, 16, 100);
        assertTrue(bbqParams.validate(validParams));
        assertEquals(64, bbqParams.getMaxConnections());
        assertEquals(256, bbqParams.getBeamWidth());

        // Invalid parameters (SQ encoder instead of BBQ)
        MethodComponentContext sqEncoder = new MethodComponentContext("sq", encoderParams);
        Map<String, Object> invalidParams = new HashMap<>();
        invalidParams.put(METHOD_ENCODER_PARAMETER, sqEncoder);
        assertFalse(bbqParams.validate(invalidParams));
    }
}
