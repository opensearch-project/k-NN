/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class EmptyQuantizedByteVectorValuesTests extends KNNTestCase {

    public void testDimension_thenDelegatesToFloatVectorValues() {
        FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        when(floatVectorValues.dimension()).thenReturn(128);

        EmptyQuantizedByteVectorValues values = new EmptyQuantizedByteVectorValues(floatVectorValues);

        assertEquals(128, values.dimension());
        verify(floatVectorValues).dimension();
    }

    public void testSize_thenDelegatesToFloatVectorValues() {
        FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        when(floatVectorValues.size()).thenReturn(0);

        EmptyQuantizedByteVectorValues values = new EmptyQuantizedByteVectorValues(floatVectorValues);

        assertEquals(0, values.size());
        verify(floatVectorValues).size();
    }

    @SneakyThrows
    public void testVectorValue_thenThrowsUnsupportedOperationException() {
        EmptyQuantizedByteVectorValues values = new EmptyQuantizedByteVectorValues(mock(FloatVectorValues.class));

        expectThrows(UnsupportedOperationException.class, () -> values.vectorValue(0));
    }

    public void testIterator_thenDelegatesToFloatVectorValues() {
        FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        FloatVectorValues.DocIndexIterator expectedIterator = mock(FloatVectorValues.DocIndexIterator.class);
        when(floatVectorValues.iterator()).thenReturn(expectedIterator);

        EmptyQuantizedByteVectorValues values = new EmptyQuantizedByteVectorValues(floatVectorValues);

        assertSame(expectedIterator, values.iterator());
        verify(floatVectorValues).iterator();
    }

    @SneakyThrows
    public void testCopy_thenWrapsCopiedFloatVectorValues() {
        FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        FloatVectorValues copiedFloatVectorValues = mock(FloatVectorValues.class);
        when(floatVectorValues.copy()).thenReturn(copiedFloatVectorValues);

        EmptyQuantizedByteVectorValues values = new EmptyQuantizedByteVectorValues(floatVectorValues);
        EmptyQuantizedByteVectorValues copiedValues = values.copy();

        assertNotSame(values, copiedValues);
        verify(floatVectorValues).copy();
    }

    public void testGetEncoding_thenReturnsByte() {
        EmptyQuantizedByteVectorValues values = new EmptyQuantizedByteVectorValues(mock(FloatVectorValues.class));

        assertEquals(VectorEncoding.BYTE, values.getEncoding());
    }

    @SneakyThrows
    public void testQuantizationSpecificMethods_thenReturnEmptyDefaults() {
        EmptyQuantizedByteVectorValues values = new EmptyQuantizedByteVectorValues(mock(FloatVectorValues.class));

        assertNull(values.scorer(new float[] { 1.0f }));
        assertNull(values.getSlice());
        assertNull(values.getCorrectiveTerms(0));
        assertNull(values.getQuantizer());
        assertNull(values.getScalarEncoding());
        assertNull(values.getCentroid());
        assertEquals(0f, values.getCentroidDP(), 0f);
    }
}
