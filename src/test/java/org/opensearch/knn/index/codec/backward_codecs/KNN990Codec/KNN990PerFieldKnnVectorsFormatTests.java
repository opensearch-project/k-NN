/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN990Codec;

import junit.framework.TestCase;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.codec.params.KNNBBQVectorsFormatParams;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

public class KNN990PerFieldKnnVectorsFormatTests extends TestCase {

    private MapperService mockMapperService;
    private KNN990PerFieldKnnVectorsFormat format;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mockMapperService = Mockito.mock(MapperService.class);
        format = new KNN990PerFieldKnnVectorsFormat(Optional.of(mockMapperService));
    }

    public void testConstructor_whenCalled_thenFormatCreated() {
        assertNotNull(format);
    }

    public void testBBQVectorsFormatParams_whenCalled_thenReturnValidParams() {
        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_BBQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderContext);
        params.put(METHOD_PARAMETER_M, 16);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, 100);

        KNNBBQVectorsFormatParams bbqParams = new KNNBBQVectorsFormatParams(params, 16, 100);

        assertTrue(bbqParams.validate(params));
        assertEquals(16, bbqParams.getMaxConnections());
        assertEquals(100, bbqParams.getBeamWidth());
        assertTrue(bbqParams.isBBQEnabled());

        // Test validation with invalid encoder
        MethodComponentContext invalidEncoder = new MethodComponentContext("invalid", encoderParams);
        Map<String, Object> invalidParams = new HashMap<>();
        invalidParams.put(METHOD_ENCODER_PARAMETER, invalidEncoder);
        assertFalse(bbqParams.validate(invalidParams));
    }
}
