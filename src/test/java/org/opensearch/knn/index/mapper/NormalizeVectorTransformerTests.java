/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;

public class NormalizeVectorTransformerTests extends KNNTestCase {
    private final NormalizeVectorTransformer transformer = new NormalizeVectorTransformer();
    private static final float DELTA = 0.001f; // Delta for floating point comparisons

    public void testNormalizeTransformer_withNullVector_thenThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> transformer.transform((float[]) null));
    }

    public void testNormalizeTransformer_withEmptyVector_thenThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> transformer.transform(new float[0]));
    }

    public void testNormalizeTransformer_withByteVector_thenThrowsException() {
        assertThrows(UnsupportedOperationException.class, () -> transformer.transform(new byte[0]));
    }

    public void testNormalizeTransformer_withValidVector_thenSuccess() {
        float[] input = { -3.0f, 4.0f };
        transformer.transform(input);

        assertEquals(-0.6f, input[0], DELTA);
        assertEquals(0.8f, input[1], DELTA);

        // Verify the magnitude is 1
        assertEquals(1.0f, calculateMagnitude(input), DELTA);
    }

    private float calculateMagnitude(float[] vector) {
        float magnitude = 0.0f;
        for (float value : vector) {
            magnitude += value * value;
        }
        return (float) Math.sqrt(magnitude);
    }

}
