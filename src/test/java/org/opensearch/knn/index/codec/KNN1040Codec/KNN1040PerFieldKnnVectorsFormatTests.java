/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.KNNTestCase;

public class KNN1040PerFieldKnnVectorsFormatTests extends KNNTestCase {

    public void testToTinySegmentsThreshold_whenNegativeOne_thenReturnsIntegerMaxValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(-1));
    }

    public void testToTinySegmentsThreshold_whenZero_thenReturnsZero() {
        // 0 → 0 → docCount < 0 is never true → always build the graph (matches Faiss semantics).
        assertEquals(0, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(0));
    }

    public void testToTinySegmentsThreshold_whenPositive_thenReturnsSameValue() {
        assertEquals(500, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(500));
        assertEquals(100, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(100));
        assertEquals(1, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(1));
    }

    public void testToTinySegmentsThreshold_whenLargeNegative_thenReturnsIntegerMaxValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(Integer.MIN_VALUE));
    }

    public void testToTinySegmentsThreshold_whenIntegerMaxValue_thenReturnsSameValue() {
        assertEquals(Integer.MAX_VALUE, KNN1040PerFieldKnnVectorsFormat.toTinySegmentsThreshold(Integer.MAX_VALUE));
    }
}
