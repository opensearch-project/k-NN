/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.KNNTestCase;

public class Faiss1040HalfFloatScalarQuantizedKnnVectorsFormatTests extends KNNTestCase {

    public void testGetName_whenHalfFloatSqFormat_thenDistinctFromFloatSqFormat() {
        String floatName = new Faiss1040ScalarQuantizedKnnVectorsFormat().getName();
        String halfFloatName = new Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat().getName();
        assertEquals("Faiss1040ScalarQuantizedKnnVectorsFormat", floatName);
        assertEquals("Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat", halfFloatName);
        assertNotEquals("SPI names must differ or the codec cannot resolve both formats", floatName, halfFloatName);
    }

    public void testHalfFloatSqFormat_thenFlatDelegateIsHalfFloatTyped() {
        // The 16x case must write an fp16 .vec, not an fp32 one.
        assertNotNull(new Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat().getFaissSqFlatFormat());
        assertNotEquals(
            new Faiss1040ScalarQuantizedKnnVectorsFormat().getFaissSqFlatFormat().toString(),
            new Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat().getFaissSqFlatFormat().toString()
        );
    }
}
