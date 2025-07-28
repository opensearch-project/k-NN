/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import junit.framework.TestCase;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;

public class LuceneBBQEncoderTests extends TestCase {

    public void testGetMethodComponent_whenCalled_thenReturnConsistentComponent() {
        LuceneBBQEncoder encoder1 = new LuceneBBQEncoder();
        LuceneBBQEncoder encoder2 = new LuceneBBQEncoder();

        MethodComponent component1 = encoder1.getMethodComponent();
        MethodComponent component2 = encoder2.getMethodComponent();

        assertNotNull(component1);
        assertEquals(ENCODER_BBQ, component1.getName());
        assertSame(component1, component2);
    }

    public void testCalculateCompressionLevel_whenCalled_thenReturnNotConfigured() {
        LuceneBBQEncoder encoder = new LuceneBBQEncoder();

        // Test with null parameters
        assertEquals(CompressionLevel.NOT_CONFIGURED, encoder.calculateCompressionLevel(null, null));

        // Test with empty context
        MethodComponentContext emptyContext = new MethodComponentContext(ENCODER_BBQ, new HashMap<>());
        assertEquals(CompressionLevel.NOT_CONFIGURED, encoder.calculateCompressionLevel(emptyContext, null));

        // Test with populated context
        Map<String, Object> params = new HashMap<>();
        params.put("param1", "value1");
        params.put("param2", 42);
        MethodComponentContext populatedContext = new MethodComponentContext(ENCODER_BBQ, params);
        assertEquals(CompressionLevel.NOT_CONFIGURED, encoder.calculateCompressionLevel(populatedContext, null));
    }

    public void testGetName_whenCalled_thenReturnEncoderBBQ() {
        LuceneBBQEncoder encoder = new LuceneBBQEncoder();
        assertEquals(ENCODER_BBQ, encoder.getName());
    }
}
