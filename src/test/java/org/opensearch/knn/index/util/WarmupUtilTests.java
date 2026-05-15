/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link WarmupUtil}.
 * <p>
 * Verifies that each {@code readAll} overload correctly warms up vector data:
 * <ul>
 *   <li>{@link IndexInput} — seeks to 0 and reads every byte sequentially.</li>
 *   <li>{@link FloatVectorValues} / {@link ByteVectorValues} — delegates to the backing
 *       {@link IndexInput} when the values implement {@link HasIndexSlice}, or falls back
 *       to iterating each vector via {@code vectorValue(i)}.</li>
 * </ul>
 */
public class WarmupUtilTests extends KNNTestCase {

    // Verify that readAll(IndexInput) seeks to the beginning and reads every byte.
    public void testReadAllIndexInput_readsAllBytes() throws IOException {
        IndexInput mockInput = mock(IndexInput.class);
        long length = 5L;
        when(mockInput.length()).thenReturn(length);

        WarmupUtil.readAll(mockInput);

        // Should start from position 0 and read exactly `length` bytes
        verify(mockInput).seek(0);
        verify(mockInput, times((int) length)).readByte();
    }

    // When FloatVectorValues does NOT implement HasIndexSlice, readAll should
    // iterate through every ordinal and call vectorValue(i) for each one.
    public void testReadAllFloatVectorValues_whenNotHasIndexSlice_readsAllValues() throws IOException {
        int size = 3;
        FloatVectorValues mockValues = mock(FloatVectorValues.class);
        when(mockValues.size()).thenReturn(size);

        WarmupUtil.readAll(mockValues);

        for (int i = 0; i < size; i++) {
            verify(mockValues).vectorValue(i);
        }
    }

    // When ByteVectorValues does NOT implement HasIndexSlice, readAll should
    // iterate through every ordinal and call vectorValue(i) for each one.
    public void testReadAllByteVectorValues_whenNotHasIndexSlice_readsAllValues() throws IOException {
        int size = 4;
        ByteVectorValues mockValues = mock(ByteVectorValues.class);
        when(mockValues.size()).thenReturn(size);

        WarmupUtil.readAll(mockValues);

        for (int i = 0; i < size; i++) {
            verify(mockValues).vectorValue(i);
        }
    }

    // When FloatVectorValues implements HasIndexSlice, readAll should delegate
    // to readAll(IndexInput) using the slice, reading all bytes from it.
    public void testReadAllFloatVectorValues_whenHasIndexSlice_readsViaSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        long sliceLength = 10L;
        when(mockSlice.length()).thenReturn(sliceLength);

        FloatVectorValuesWithSlice mockValues = mock(FloatVectorValuesWithSlice.class);
        when(mockValues.getSlice()).thenReturn(mockSlice);

        WarmupUtil.readAll((FloatVectorValues) mockValues);

        // Warmup should go through the slice, not individual vectorValue calls
        verify(mockSlice).seek(0);
        verify(mockSlice, times((int) sliceLength)).readByte();
    }

    // When ByteVectorValues implements HasIndexSlice, readAll should delegate
    // to readAll(IndexInput) using the slice, reading all bytes from it.
    public void testReadAllByteVectorValues_whenHasIndexSlice_readsViaSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        long sliceLength = 8L;
        when(mockSlice.length()).thenReturn(sliceLength);

        ByteVectorValuesWithSlice mockValues = mock(ByteVectorValuesWithSlice.class);
        when(mockValues.getSlice()).thenReturn(mockSlice);

        WarmupUtil.readAll((ByteVectorValues) mockValues);

        // Warmup should go through the slice, not individual vectorValue calls
        verify(mockSlice).seek(0);
        verify(mockSlice, times((int) sliceLength)).readByte();
    }

    // Abstract helper types that combine VectorValues with HasIndexSlice so
    // Mockito can produce mocks matching the instanceof check in WarmupUtil.
    abstract static class FloatVectorValuesWithSlice extends FloatVectorValues implements HasIndexSlice {}

    abstract static class ByteVectorValuesWithSlice extends ByteVectorValues implements HasIndexSlice {}

    // Verify that readAll(FloatVectorValues) throws NullPointerException when passed null.
    public void testReadAllFloatVectorValues_whenNull_throwsNullPointerException() {
        expectThrows(NullPointerException.class, () -> WarmupUtil.readAll((FloatVectorValues) null));
    }

    // Verify that readAll(ByteVectorValues) throws NullPointerException when passed null.
    public void testReadAllByteVectorValues_whenNull_throwsNullPointerException() {
        expectThrows(NullPointerException.class, () -> WarmupUtil.readAll((ByteVectorValues) null));
    }

    // Verify that readAll(IndexInput) throws NullPointerException when passed null.
    public void testReadAllIndexInput_whenNull_throwsNullPointerException() {
        expectThrows(NullPointerException.class, () -> WarmupUtil.readAll((IndexInput) null));
    }
}
