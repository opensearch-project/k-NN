/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.opensearch.knn.KNNTestCase;

import java.util.concurrent.Executors;

public class KNN9120HnswBinaryVectorsFormatTests extends KNNTestCase {

    public void testConstructor_whenDefault_thenSucceeds() {
        KNN9120HnswBinaryVectorsFormat format = new KNN9120HnswBinaryVectorsFormat();
        assertNotNull(format);
    }

    public void testConstructor_whenMaxConnAndBeamWidthProvided_thenSucceeds() {
        KNN9120HnswBinaryVectorsFormat format = new KNN9120HnswBinaryVectorsFormat(16, 100);
        assertNotNull(format);
        String str = format.toString();
        assertTrue(str.contains("maxConn"));
        assertTrue(str.contains("beamWidth"));
    }

    public void testConstructor_whenMergeWorkerAndNoExecutor_thenSucceeds() {
        KNN9120HnswBinaryVectorsFormat format = new KNN9120HnswBinaryVectorsFormat(16, 100, 1, null);
        assertNotNull(format);
    }

    public void testConstructor_whenCustomTinySegmentsThreshold_thenSucceeds() {
        KNN9120HnswBinaryVectorsFormat format = new KNN9120HnswBinaryVectorsFormat(16, 100, 1, null, 500);
        assertNotNull(format);
    }

    public void testConstructor_whenMaxValueTinySegmentsThreshold_thenSucceeds() {
        KNN9120HnswBinaryVectorsFormat format = new KNN9120HnswBinaryVectorsFormat(16, 100, 1, null, Integer.MAX_VALUE);
        assertNotNull(format);
    }

    public void testConstructor_whenZeroTinySegmentsThreshold_thenSucceeds() {
        KNN9120HnswBinaryVectorsFormat format = new KNN9120HnswBinaryVectorsFormat(16, 100, 1, null, 0);
        assertNotNull(format);
    }

    public void testConstructor_whenInvalidMaxConn_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> new KNN9120HnswBinaryVectorsFormat(0, 100, 1, null, 0));
    }

    public void testConstructor_whenInvalidBeamWidth_thenThrows() {
        expectThrows(IllegalArgumentException.class, () -> new KNN9120HnswBinaryVectorsFormat(16, 0, 1, null, 0));
    }

    public void testConstructor_whenSingleWorkerWithExecutor_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN9120HnswBinaryVectorsFormat(16, 100, 1, Executors.newFixedThreadPool(1), 0)
        );
    }
}
