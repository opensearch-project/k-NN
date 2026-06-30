/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.util;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.util.QuantizationUtils.FloatArrayWrapper;

import java.io.IOException;

public class QuantizationUtilsTests extends KNNTestCase {

    public void testFloatArrayWrapperConstructorAndGetter() {
        float[] array = { 1.0f, 2.5f, 3.7f };
        FloatArrayWrapper wrapper = new FloatArrayWrapper(array);

        assertArrayEquals(array, wrapper.getArray(), 0.0f);
    }

    public void testFloatArrayWrapperEmptyArray() {
        float[] array = {};
        FloatArrayWrapper wrapper = new FloatArrayWrapper(array);

        assertNotNull(wrapper.getArray());
        assertEquals(0, wrapper.getArray().length);
    }

    public void testFloatArrayWrapperSerializationRoundTrip() throws IOException {
        float[] array = { 0.1f, -2.5f, 100.0f, 0.0f };
        FloatArrayWrapper original = new FloatArrayWrapper(array);

        BytesStreamOutput out = new BytesStreamOutput();
        original.writeTo(out);

        FloatArrayWrapper deserialized = new FloatArrayWrapper(out.bytes().streamInput());
        assertArrayEquals(array, deserialized.getArray(), 0.0f);
    }

    public void testFloatArrayWrapperSerializationWithSingleElement() throws IOException {
        float[] array = { 42.0f };
        FloatArrayWrapper original = new FloatArrayWrapper(array);

        BytesStreamOutput out = new BytesStreamOutput();
        original.writeTo(out);

        FloatArrayWrapper deserialized = new FloatArrayWrapper(out.bytes().streamInput());
        assertArrayEquals(array, deserialized.getArray(), 0.0f);
    }

    public void testFloatArrayWrapperOptionalArrayRoundTrip() throws IOException {
        float[] array = { 1.0f, 2.0f, 3.0f };
        FloatArrayWrapper[] wrappers = new FloatArrayWrapper[] { new FloatArrayWrapper(array) };

        BytesStreamOutput out = new BytesStreamOutput();
        out.writeOptionalArray(wrappers);

        FloatArrayWrapper[] deserialized = out.bytes().streamInput().readOptionalArray(FloatArrayWrapper::new, FloatArrayWrapper[]::new);

        assertNotNull(deserialized);
        assertEquals(1, deserialized.length);
        assertArrayEquals(array, deserialized[0].getArray(), 0.0f);
    }

    public void testFloatArrayWrapperOptionalArrayNull() throws IOException {
        BytesStreamOutput out = new BytesStreamOutput();
        out.writeOptionalArray(null);

        FloatArrayWrapper[] deserialized = out.bytes().streamInput().readOptionalArray(FloatArrayWrapper::new, FloatArrayWrapper[]::new);

        assertNull(deserialized);
    }

    public void testFloatArrayWrapperWithNullArray_getterReturnsNull() {
        FloatArrayWrapper wrapper = new FloatArrayWrapper((float[]) null);

        assertNull(wrapper.getArray());
    }

    public void testFloatArrayWrapperWithNullArray_writeToThrowsNullPointerException() {
        FloatArrayWrapper wrapper = new FloatArrayWrapper((float[]) null);

        NullPointerException exception = assertThrows(NullPointerException.class, () -> wrapper.writeTo(new BytesStreamOutput()));
        assertTrue(exception.getMessage().contains("values"));
    }
}
