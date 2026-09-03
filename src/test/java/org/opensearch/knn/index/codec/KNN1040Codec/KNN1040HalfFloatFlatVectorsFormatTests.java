/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

public class KNN1040HalfFloatFlatVectorsFormatTests extends KNNTestCase {

    public void testFormat_whenCreated_thenSimpleNameMatches() {
        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        assertEquals("KNN1040HalfFloatFlatVectorsFormat", format.getClass().getSimpleName());
    }

    public void testGetMaxDimensions_whenCalled_thenReturnsLuceneMax() {
        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("test-field"));
    }

    public void testToString_whenCalled_thenContainsScorerInfo() {
        KNN1040HalfFloatFlatVectorsFormat format = new KNN1040HalfFloatFlatVectorsFormat();
        String str = format.toString();
        assertTrue(str.contains("KNN1040HalfFloatFlatVectorsFormat"));
        assertTrue("toString should expose scorer info, was: " + str, str.contains("scorer="));
    }
}
