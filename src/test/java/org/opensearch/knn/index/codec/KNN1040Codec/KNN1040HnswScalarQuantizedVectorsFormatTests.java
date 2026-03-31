/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.concurrent.Executors;

import static org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN1040HnswScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testDefaultConstructor() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat();
        assertNotNull(format);
        String str = format.toString();
        assertTrue(str.contains("KNN1040HnswScalarQuantizedVectorsFormat"));
        assertTrue(str.contains(SINGLE_BIT_QUERY_NIBBLE.name()));
    }

    public void testCustomConstructor() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE,
            32,
            200,
            1,
            null
        );
        assertNotNull(format);
        assertEquals("KNN1040HnswScalarQuantizedVectorsFormat", format.getName());
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetMaxDimensions_returnsLuceneMax() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetName_returnsClassName() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat();
        assertEquals("KNN1040HnswScalarQuantizedVectorsFormat", format.getName());
    }

    public void testConstructor_invalidMaxConn_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 0, 100, 1, null)
        );
    }

    public void testConstructor_invalidBeamWidth_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 16, 0, 1, null)
        );
    }

    public void testConstructor_singleWorkerWithExecutor_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 16, 100, 1, Executors.newFixedThreadPool(1))
        );
    }

    public void testConstructor_allEncodings() {
        for (ScalarEncoding encoding : ScalarEncoding.values()) {
            KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat(encoding, 16, 100, 1, null);
            assertNotNull(format);
            assertTrue(format.toString().contains(encoding.name()));
        }
    }
}
