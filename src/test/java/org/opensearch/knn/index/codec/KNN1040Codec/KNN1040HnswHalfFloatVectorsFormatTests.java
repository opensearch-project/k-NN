/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class KNN1040HnswHalfFloatVectorsFormatTests extends KNNTestCase {

    public void testConstructor_whenDefault_thenSucceeds() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat();
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatVectorsFormat", format.getName());
    }

    public void testConstructor_whenMaxConnAndBeamWidthProvided_thenSucceeds() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat(32, 200, 1, null);
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatVectorsFormat", format.getName());
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetMaxDimensions_whenCalled_thenReturnsLuceneMax() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testToString_whenCalled_thenContainsParams() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat(32, 200, 1, null, 500);
        String str = format.toString();
        assertTrue(str.contains("KNN1040HnswHalfFloatVectorsFormat"));
        assertTrue(str.contains("maxConn=32"));
        assertTrue(str.contains("beamWidth=200"));
        assertTrue(str.contains("tinySegmentsThreshold=500"));
        assertTrue(str.contains("KNN1040HalfFloatFlatVectorsFormat"));
    }

    public void testConstructor_whenInvalidMaxConn_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> new KNN1040HnswHalfFloatVectorsFormat(0, 100, 1, null));
    }

    public void testConstructor_whenMaxConnExceedsMaximum_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> new KNN1040HnswHalfFloatVectorsFormat(513, 100, 1, null));
    }

    public void testConstructor_whenInvalidBeamWidth_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> new KNN1040HnswHalfFloatVectorsFormat(16, 0, 1, null));
    }

    public void testConstructor_whenBeamWidthExceedsMaximum_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> new KNN1040HnswHalfFloatVectorsFormat(16, 3201, 1, null));
    }

    public void testConstructor_whenSingleWorkerWithExecutor_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswHalfFloatVectorsFormat(16, 100, 1, Executors.newFixedThreadPool(1))
        );
    }

    public void testConstructor_whenMultipleWorkersWithExecutor_thenSucceeds() {
        ExecutorService mergeExec = Executors.newFixedThreadPool(2);
        try {
            KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat(16, 100, 2, mergeExec);
            assertNotNull(format);
            assertEquals("KNN1040HnswHalfFloatVectorsFormat", format.getName());
        } finally {
            mergeExec.shutdownNow();
        }
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

    public void testConstructor_whenMaxValueTinySegmentsThreshold_thenSucceeds() {
        KNN1040HnswHalfFloatVectorsFormat format = new KNN1040HnswHalfFloatVectorsFormat(16, 100, 1, null, Integer.MAX_VALUE);
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatVectorsFormat", format.getName());
    }
}
