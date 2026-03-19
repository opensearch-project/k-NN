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

import static org.opensearch.knn.common.KNNConstants.ENCODER_OPTIMIZED_SCALAR_QUANTIZER;;

public class OptimizedScalarQuantizerEncoderTests extends TestCase {

    public void testGetMethodComponent_whenCalled_thenReturnConsistentComponent() {
        OptimizedScalarQuantizerEncoder encoder1 = new OptimizedScalarQuantizerEncoder();
        OptimizedScalarQuantizerEncoder encoder2 = new OptimizedScalarQuantizerEncoder();

        MethodComponent component1 = encoder1.getMethodComponent();
        MethodComponent component2 = encoder2.getMethodComponent();

        assertNotNull(component1);
        assertEquals(ENCODER_OPTIMIZED_SCALAR_QUANTIZER, component1.getName());
        assertSame(component1, component2);
    }

    public void testCalculateCompressionLevel_whenCalled_thenReturnX32() {
        OptimizedScalarQuantizerEncoder encoder = new OptimizedScalarQuantizerEncoder();

        // Test with null parameters
        assertEquals(CompressionLevel.x32, encoder.calculateCompressionLevel(null, null));

        // Test with empty context
        MethodComponentContext emptyContext = new MethodComponentContext(ENCODER_OPTIMIZED_SCALAR_QUANTIZER, new HashMap<>());
        assertEquals(CompressionLevel.x32, encoder.calculateCompressionLevel(emptyContext, null));

        // Test with populated context
        Map<String, Object> params = new HashMap<>();
        params.put("param1", "value1");
        params.put("param2", 42);
        MethodComponentContext populatedContext = new MethodComponentContext(ENCODER_OPTIMIZED_SCALAR_QUANTIZER, params);
        assertEquals(CompressionLevel.x32, encoder.calculateCompressionLevel(populatedContext, null));
    }

    public void testGetName_whenCalled_thenReturnEncoderOptimizedSQ() {
        OptimizedScalarQuantizerEncoder encoder = new OptimizedScalarQuantizerEncoder();
        assertEquals(ENCODER_OPTIMIZED_SCALAR_QUANTIZER, encoder.getName());
    }
}
