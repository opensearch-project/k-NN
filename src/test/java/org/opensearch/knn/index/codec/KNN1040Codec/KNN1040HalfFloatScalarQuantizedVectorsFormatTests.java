/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import static org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN1040HalfFloatScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testConstructor_withHalfFloatRawVectorDataType_doesNotThrow() {
        KNN1040HalfFloatScalarQuantizedVectorsFormat format = new KNN1040HalfFloatScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        assertNotNull(format);
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testDefaultConstructor_usesSingleBitQueryNibble() {
        KNN1040HalfFloatScalarQuantizedVectorsFormat format = new KNN1040HalfFloatScalarQuantizedVectorsFormat();
        assertTrue(format.toString().contains(SINGLE_BIT_QUERY_NIBBLE.name()));
    }

    public void testGetName_returnsClassName() {
        KNN1040HalfFloatScalarQuantizedVectorsFormat format = new KNN1040HalfFloatScalarQuantizedVectorsFormat();
        assertEquals("KNN1040HalfFloatScalarQuantizedVectorsFormat", format.getName());
    }

    public void testGetName_floatAndHalfFloatVariantsAreDistinct() {
        KNN1040ScalarQuantizedVectorsFormat floatFormat = new KNN1040ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        KNN1040HalfFloatScalarQuantizedVectorsFormat halfFloatFormat = new KNN1040HalfFloatScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE
        );
        assertNotEquals(
            "FLOAT and HALF_FLOAT SQ 1-bit formats must have different names or read-time SPI reconstruction cannot tell them apart",
            floatFormat.getName(),
            halfFloatFormat.getName()
        );
    }
}
