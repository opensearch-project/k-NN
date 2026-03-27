/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class ScalarQuantizedFloatVectorValuesWithIndexInputSliceTests extends KNNTestCase {

    @SneakyThrows
    public void testDimension_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.dimension()).thenReturn(128);
        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(fvv, mock(QuantizedByteVectorValues.class));
        assertEquals(128, wrapper.dimension());
        verify(fvv).dimension();
    }

    @SneakyThrows
    public void testSize_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.size()).thenReturn(400);
        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(fvv, mock(QuantizedByteVectorValues.class));
        assertEquals(400, wrapper.size());
        verify(fvv).size();
    }

    @SneakyThrows
    public void testVectorValue_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        float[] expected = { 1.0f, 2.0f, 3.0f };
        when(fvv.vectorValue(5)).thenReturn(expected);
        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(fvv, mock(QuantizedByteVectorValues.class));
        assertSame(expected, wrapper.vectorValue(5));
        verify(fvv).vectorValue(5);
    }

    @SneakyThrows
    public void testCopy_thenReturnsNewWrappedInstance() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        FloatVectorValues fvvCopy = mock(FloatVectorValues.class);
        QuantizedByteVectorValues qbvv = mock(QuantizedByteVectorValues.class);
        QuantizedByteVectorValues qbvvCopy = mock(QuantizedByteVectorValues.class);
        when(fvv.copy()).thenReturn(fvvCopy);
        when(qbvv.copy()).thenReturn(qbvvCopy);

        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(fvv, qbvv);
        FloatVectorValues copied = wrapper.copy();

        assertNotSame(wrapper, copied);
        assertTrue(copied instanceof ScalarQuantizedFloatVectorValuesWithIndexInputSlice);
        verify(fvv).copy();
        verify(qbvv).copy();
    }

    @SneakyThrows
    public void testGetEncoding_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.getEncoding()).thenReturn(VectorEncoding.FLOAT32);
        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(fvv, mock(QuantizedByteVectorValues.class));
        assertEquals(VectorEncoding.FLOAT32, wrapper.getEncoding());
        verify(fvv).getEncoding();
    }

    @SneakyThrows
    public void testGetSlice_thenReturnsSliceFromQuantizedValues() {
        QuantizedByteVectorValues qbvv = mock(QuantizedByteVectorValues.class);
        IndexInput expectedSlice = mock(IndexInput.class);
        when(qbvv.getSlice()).thenReturn(expectedSlice);

        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(mock(FloatVectorValues.class), qbvv);
        assertSame(expectedSlice, wrapper.getSlice());
        verify(qbvv).getSlice();
    }

    @SneakyThrows
    public void testImplementsHasIndexSlice() {
        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(
            mock(FloatVectorValues.class),
            mock(QuantizedByteVectorValues.class)
        );
        assertTrue(wrapper instanceof HasIndexSlice);
    }

    @SneakyThrows
    public void testIterator_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        FloatVectorValues.DocIndexIterator expectedIterator = mock(FloatVectorValues.DocIndexIterator.class);
        when(fvv.iterator()).thenReturn(expectedIterator);

        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(fvv, mock(QuantizedByteVectorValues.class));
        assertSame(expectedIterator, wrapper.iterator());
        verify(fvv).iterator();
    }

    @SneakyThrows
    public void testScorer_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        VectorScorer expectedScorer = mock(VectorScorer.class);
        float[] target = { 1.0f, 2.0f };
        when(fvv.scorer(target)).thenReturn(expectedScorer);

        var wrapper = new ScalarQuantizedFloatVectorValuesWithIndexInputSlice(fvv, mock(QuantizedByteVectorValues.class));
        assertSame(expectedScorer, wrapper.scorer(target));
        verify(fvv).scorer(target);
    }
}
