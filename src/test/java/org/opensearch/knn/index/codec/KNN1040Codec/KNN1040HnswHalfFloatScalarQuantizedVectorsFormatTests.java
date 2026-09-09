/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import static org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN1040HnswHalfFloatScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testDefaultConstructor() {
        KNN1040HnswHalfFloatScalarQuantizedVectorsFormat format = new KNN1040HnswHalfFloatScalarQuantizedVectorsFormat();
        assertNotNull(format);
        String str = format.toString();
        assertTrue(str.contains("KNN1040HnswHalfFloatScalarQuantizedVectorsFormat"));
        assertTrue(str.contains(SINGLE_BIT_QUERY_NIBBLE.name()));
    }

    public void testCustomConstructor() {
        KNN1040HnswHalfFloatScalarQuantizedVectorsFormat format = new KNN1040HnswHalfFloatScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE,
            32,
            200,
            1,
            null
        );
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatScalarQuantizedVectorsFormat", format.getName());
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testConstructor_whenCustomTinySegmentsThreshold_thenSucceeds() {
        KNN1040HnswHalfFloatScalarQuantizedVectorsFormat format = new KNN1040HnswHalfFloatScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE,
            16,
            100,
            1,
            null,
            500
        );
        assertNotNull(format);
        assertEquals("KNN1040HnswHalfFloatScalarQuantizedVectorsFormat", format.getName());
    }

    public void testConstructor_invalidMaxConn_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswHalfFloatScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 0, 100, 1, null)
        );
    }

    public void testGetName_floatAndHalfFloatVariantsAreDistinct() {
        KNN1040HnswScalarQuantizedVectorsFormat floatFormat = new KNN1040HnswScalarQuantizedVectorsFormat();
        KNN1040HnswHalfFloatScalarQuantizedVectorsFormat halfFloatFormat = new KNN1040HnswHalfFloatScalarQuantizedVectorsFormat();
        assertNotEquals(
            "FLOAT and HALF_FLOAT HNSW SQ 1-bit formats must have different names or read-time SPI "
                + "reconstruction cannot tell them apart",
            floatFormat.getName(),
            halfFloatFormat.getName()
        );
    }

    public void testSpiReconstruction_resolvesToHalfFloatVariant() {
        KNN1040HnswHalfFloatScalarQuantizedVectorsFormat writeFormat = new KNN1040HnswHalfFloatScalarQuantizedVectorsFormat();
        KnnVectorsFormat readFormat = KnnVectorsFormat.forName(writeFormat.getName());
        assertTrue(
            "forName(" + writeFormat.getName() + ") should resolve to the half_float HNSW SQ format",
            readFormat instanceof KNN1040HnswHalfFloatScalarQuantizedVectorsFormat
        );
    }
}
