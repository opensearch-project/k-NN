/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.concurrent.Executors;

public class KNN1040HnswHalfFloatVectorsFormatTests extends KNNTestCase {

    public void testDefaultConstructor() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat();
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatVectorsFormat", format.getName());
    }

    public void testCustomConstructor() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat(32, 200, 1, null);
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatVectorsFormat", format.getName());
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetMaxDimensions_returnsLuceneMax() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testToString_containsParams() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat(32, 200, 1, null);
        String str = format.toString();
        assertTrue(str.contains("KNN1040HnswHalfFloatVectorsFormat"));
        assertTrue(str.contains("maxConn=32"));
        assertTrue(str.contains("beamWidth=200"));
        assertTrue(str.contains("KNN1040HalfFloatFlatVectorsFormat"));
    }

    public void testConstructor_invalidMaxConn_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> new KNN1040HnswHalfFloatVectorsFormat(0, 100, 1, null));
    }

    public void testConstructor_invalidBeamWidth_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> new KNN1040HnswHalfFloatVectorsFormat(16, 0, 1, null));
    }

    public void testConstructor_singleWorkerWithExecutor_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswHalfFloatVectorsFormat(16, 100, 1, Executors.newFixedThreadPool(1))
        );
    }

    public void testConstructor_whenCustomTinySegmentsThreshold_thenSucceeds() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat(16, 100, 1, null, 500);
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatVectorsFormat", format.getName());
    }

    public void testConstructor_whenZeroTinySegmentsThreshold_thenSucceeds() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat(16, 100, 1, null, 0);
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatVectorsFormat", format.getName());
    }
}
