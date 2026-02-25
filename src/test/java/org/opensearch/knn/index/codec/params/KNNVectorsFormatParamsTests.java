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

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

public class KNNVectorsFormatParamsTests extends TestCase {
    private static final int DEFAULT_MAX_CONNECTIONS = 16;
    private static final int DEFAULT_BEAM_WIDTH = 100;

    public void testInitParams_whenCalled_thenReturnDefaultParams() {
        KNNVectorsFormatParams knnVectorsFormatParams = new KNNVectorsFormatParams(
            new HashMap<>(),
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );
        assertEquals(DEFAULT_MAX_CONNECTIONS, knnVectorsFormatParams.getMaxConnections());
        assertEquals(DEFAULT_BEAM_WIDTH, knnVectorsFormatParams.getBeamWidth());
    }

    public void testInitParams_whenCalled_thenReturnParams() {
        int m = 64;
        int efConstruction = 128;
        Map<String, Object> params = new HashMap<>();
        params.put(METHOD_PARAMETER_M, m);
        params.put(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction);

        KNNVectorsFormatParams knnVectorsFormatParams = new KNNVectorsFormatParams(params, DEFAULT_MAX_CONNECTIONS, DEFAULT_BEAM_WIDTH);
        assertEquals(m, knnVectorsFormatParams.getMaxConnections());
        assertEquals(efConstruction, knnVectorsFormatParams.getBeamWidth());
    }

    public void testValidate_whenCalled_thenReturnTrue() {
        KNNVectorsFormatParams knnVectorsFormatParams = new KNNVectorsFormatParams(
            new HashMap<>(),
            DEFAULT_MAX_CONNECTIONS,
            DEFAULT_BEAM_WIDTH
        );
        assertTrue(knnVectorsFormatParams.validate(new HashMap<>()));
    }
}
