/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizationState;

import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateCache;

public class QuantizationStateCacheTests extends KNNTestCase {
    @SneakyThrows
    public void testQuantizationStateCache() {
        String fieldName1 = "test-field-1";
        String fieldName2 = "test-field-2";
        String fieldName3 = "test-field-3";
        float[] floatArray1 = { 1.2f, 2.3f, 3.4f };
        float[] floatArray2 = { 2.3f, 3.4f, 4.5f };
        float[] floatArray3 = { 3.4f, 4.5f, 5.6f };
        QuantizationState qs1 = new OneBitScalarQuantizationState(new SQParams(ONE_BIT), floatArray1);
        QuantizationState qs2 = new OneBitScalarQuantizationState(new SQParams(ONE_BIT), floatArray2);
        QuantizationState qs3 = new OneBitScalarQuantizationState(new SQParams(ONE_BIT), floatArray3);

        // Add test quantization states
        QuantizationStateCache.getInstance().addQuantizationState(fieldName1, qs1);
        QuantizationStateCache.getInstance().addQuantizationState(fieldName2, qs2);
        QuantizationStateCache.getInstance().addQuantizationState(fieldName3, qs3);

        // Assert all states are present
        assertEquals(qs1, QuantizationStateCache.getInstance().getQuantizationState(fieldName1));
        assertEquals(qs2, QuantizationStateCache.getInstance().getQuantizationState(fieldName2));
        assertEquals(qs3, QuantizationStateCache.getInstance().getQuantizationState(fieldName3));

        // Remove one state
        QuantizationStateCache.getInstance().evict(fieldName1);

        // Assert state has been removed, others are still present
        assertNull(QuantizationStateCache.getInstance().getQuantizationState(fieldName1));
        assertEquals(qs2, QuantizationStateCache.getInstance().getQuantizationState(fieldName2));
        assertEquals(qs3, QuantizationStateCache.getInstance().getQuantizationState(fieldName3));

        // Clear cache
        QuantizationStateCache.getInstance().clear();

        // Assert all states have been removed
        assertNull(QuantizationStateCache.getInstance().getQuantizationState(fieldName2));
        assertNull(QuantizationStateCache.getInstance().getQuantizationState(fieldName3));
    }
}
