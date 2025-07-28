/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import junit.framework.TestCase;
import org.junit.Assert;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

public class KNNBBQVectorsFormatTests extends TestCase {
    private static final int DEFAULT_MAX_CONNECTIONS = 16;
    private static final int DEFAULT_BEAM_WIDTH = 100;

    public void testInitParams_whenCalled_thenReturnDefaultParams() {
        KNNBBQVectorsFormatParams knnBBQVectorsFormatParams = new KNNBBQVectorsFormatParams(
            getDefaultParamsForConstructor(),
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        assertEquals(DEFAULT_MAX_CONNECTIONS, knnBBQVectorsFormatParams.getMaxConnections());
        assertEquals(DEFAULT_BEAM_WIDTH, knnBBQVectorsFormatParams.getBeamWidth());
        assertTrue(knnBBQVectorsFormatParams.isBBQEnabled());
    }

    public void testInitParams_whenCalled_thenReturnParams() {
        int m = 64;
        int efConstruction = 128;

        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_BBQ, encoderParams);

        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
        params.put(METHOD_PARAMETER_M, m);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction);

        KNNBBQVectorsFormatParams knnBBQVectorsFormatParams = new KNNBBQVectorsFormatParams(
            params,
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        assertEquals(m, knnBBQVectorsFormatParams.getMaxConnections());
        assertEquals(efConstruction, knnBBQVectorsFormatParams.getBeamWidth());
        assertTrue(knnBBQVectorsFormatParams.isBBQEnabled());
    }

    public void testValidate_whenCalled_thenReturnTrue() {
        Map<String, Object> params = getDefaultParamsForConstructor();
        KNNBBQVectorsFormatParams knnBBQVectorsFormatParams = new KNNBBQVectorsFormatParams(
            params,
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );
        assertTrue(knnBBQVectorsFormatParams.validate(params));
    }

    public void testValidate_whenCalled_thenReturnFalse() {
        KNNBBQVectorsFormatParams knnBBQVectorsFormatParams = new KNNBBQVectorsFormatParams(
            getDefaultParamsForConstructor(),
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );
        Map<String, Object> params = new HashMap<>();

        // Return false if encoder value is null
        params.put(METHOD_ENCODER_PARAMETER, null);
        assertFalse(knnBBQVectorsFormatParams.validate(params));

        // Return false if encoder value is not an instance of MethodComponentContext
        params.replace(METHOD_ENCODER_PARAMETER, "dummy string");
        assertFalse(knnBBQVectorsFormatParams.validate(params));

        // Return false if encoder name is not "binary"
        MethodComponentContext encoderComponentContext = new MethodComponentContext("invalid encoder name", new HashMap<>());
        params.replace(METHOD_ENCODER_PARAMETER, encoderComponentContext);
        assertFalse(knnBBQVectorsFormatParams.validate(params));
    }

    public void testValidate_whenNullParams_thenThrowException() {
        KNNBBQVectorsFormatParams knnBBQVectorsFormatParams = new KNNBBQVectorsFormatParams(
            getDefaultParamsForConstructor(),
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );

        Assert.assertThrows(NullPointerException.class, () -> knnBBQVectorsFormatParams.validate(null));
    }

    private Map<String, Object> getDefaultParamsForConstructor() {
        Map<String, Object> encoderParams = new HashMap<>();
        MethodComponentContext encoderComponentContext = new MethodComponentContext(ENCODER_BBQ, encoderParams);
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_ENCODER_PARAMETER, encoderComponentContext);
        return params;
    }
}
