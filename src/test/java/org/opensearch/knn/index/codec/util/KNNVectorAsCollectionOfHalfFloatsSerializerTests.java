/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.opensearch.knn.KNNTestCase;

import java.util.Random;

public class KNNVectorAsCollectionOfHalfFloatsSerializerTests extends KNNTestCase {

    Random random = new Random();
    private static final float FP16_TOLERANCE = 1e-3f;

    public void testVectorAsCollectionOfHalfFloatsSerializer() {
        float[] original = getArrayOfRandomFloats(20);
        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;

        byte[] encoded = new byte[original.length * 2];
        float[] decoded = new float[original.length];

        // serialize
        serializer.floatToByteArray(original, encoded, original.length);
        // deserialize
        serializer.byteToFloatArray(encoded, decoded, original.length, 0);

        // compare with fp16 precision tolerance
        for (int i = 0; i < original.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(original[i]));
            assertEquals(expected, decoded[i], FP16_TOLERANCE);
        }
    }

    public void testVectorSerializer_whenVectorBytesOffset_thenSuccess() {
        float[] original = getArrayOfRandomFloats(20);
        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;

        byte[] encoded = new byte[original.length * 2];
        serializer.floatToByteArray(original, encoded, original.length);

        int offset = randomInt(4);
        byte[] padded = new byte[encoded.length + 2 * offset];
        System.arraycopy(encoded, 0, padded, 2 * offset, encoded.length);

        float[] decoded = new float[original.length];
        serializer.byteToFloatArray(padded, decoded, original.length, 2 * offset);

        for (int i = 0; i < original.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(original[i]));
            assertEquals(expected, decoded[i], FP16_TOLERANCE);
        }
    }

    public void testEmptyVector() {
        float[] empty = new float[0];
        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;

        byte[] encoded = new byte[0];
        float[] decoded = new float[0];

        serializer.floatToByteArray(empty, encoded, empty.length);
        serializer.byteToFloatArray(encoded, decoded, empty.length, 0);

        assertEquals(0, decoded.length);
    }

    public void testInvalidByteStream_throwsException() {
        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;
        float[] output = new float[1];

        assertThrows(IllegalArgumentException.class, () -> serializer.byteToFloatArray(null, output, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> serializer.byteToFloatArray(new byte[3], output, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> serializer.byteToFloatArray(new byte[1], output, 1, 0));
    }

    public void testSpecialFloatValues() {
        float[] specialValues = new float[] { Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.0f, 0.0f };

        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;
        byte[] encoded = new byte[specialValues.length * 2];
        float[] decoded = new float[specialValues.length];

        serializer.floatToByteArray(specialValues, encoded, specialValues.length);
        serializer.byteToFloatArray(encoded, decoded, specialValues.length, 0);

        assertEquals(specialValues.length, decoded.length);
        assertTrue(Float.isNaN(decoded[0]));
        assertEquals(Float.POSITIVE_INFINITY, decoded[1], 0.0f);
        assertEquals(Float.NEGATIVE_INFINITY, decoded[2], 0.0f);
        assertEquals(-0.0f, decoded[3], 0.0f);
        assertEquals(0.0f, decoded[4], 0.0f);
    }

    public void testFp16PrecisionLimits() {
        float[] boundaryValues = new float[] {
            65504.0f,         // Max fp16 value
            -65504.0f,        // Min fp16 value
            6.103515625e-5f,  // Smallest positive normal fp16
            -6.103515625e-5f, // Smallest negative normal fp16
        };

        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;
        byte[] encoded = new byte[boundaryValues.length * 2];
        float[] decoded = new float[boundaryValues.length];

        serializer.floatToByteArray(boundaryValues, encoded, boundaryValues.length);
        serializer.byteToFloatArray(encoded, decoded, boundaryValues.length, 0);

        for (int i = 0; i < boundaryValues.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(boundaryValues[i]));
            assertEquals(expected, decoded[i], 0.0f);
        }
    }

    public void testSubnormalValues() {
        float[] subNormals = { 1e-8f, -1e-8f, 1e-10f, -1e-10f, 1e-20f, -1e-20f };

        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;
        byte[] encoded = new byte[subNormals.length * 2];
        float[] decoded = new float[subNormals.length];

        serializer.floatToByteArray(subNormals, encoded, subNormals.length);
        serializer.byteToFloatArray(encoded, decoded, subNormals.length, 0);

        for (int i = 0; i < subNormals.length; i++) {
            assertEquals(0.0f, decoded[i], 1e-7f); // subNormals become zero in FP16
        }
    }

    public void testOutOfRangeValues() {
        float[] tooLarge = { 1e10f, -1e10f, Float.MAX_VALUE, -Float.MAX_VALUE };

        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;
        byte[] encoded = new byte[tooLarge.length * 2];
        float[] decoded = new float[tooLarge.length];

        serializer.floatToByteArray(tooLarge, encoded, tooLarge.length);
        serializer.byteToFloatArray(encoded, decoded, tooLarge.length, 0);

        for (float val : decoded) {
            assertTrue(Float.isInfinite(val));
        }
    }

    public void testMidRangeRounding() {
        float[] values = { 0.1f, 0.33333f, 0.66667f, 1.5f, 3.14159f };
        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;
        byte[] encoded = new byte[values.length * 2];
        float[] decoded = new float[values.length];

        serializer.floatToByteArray(values, encoded, values.length);
        serializer.byteToFloatArray(encoded, decoded, values.length, 0);

        for (int i = 0; i < values.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(values[i]));
            assertEquals(expected, decoded[i], FP16_TOLERANCE);
        }
    }

    public void testFloatToByteArray_bufferSizeMismatch_throws() {
        int dim = 32;
        float[] input = new float[dim];
        byte[] undersized = new byte[dim * 2 - 1]; // 63 instead of 64
        byte[] oversized = new byte[dim * 2 + 1];  // 65 instead of 64

        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;

        IllegalArgumentException e1 = expectThrows(
            IllegalArgumentException.class,
            () -> { serializer.floatToByteArray(input, undersized, dim); }
        );
        assertEquals("Output buffer size mismatch. Must be 2x input length.", e1.getMessage());

        IllegalArgumentException e2 = expectThrows(
            IllegalArgumentException.class,
            () -> { serializer.floatToByteArray(input, oversized, dim); }
        );
        assertEquals("Output buffer size mismatch. Must be 2x input length.", e2.getMessage());
    }

    public void testLargeVector() {
        float[] large = getArrayOfRandomFloats(10000);
        KNNVectorAsCollectionOfHalfFloatsSerializer serializer = KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE;

        byte[] encoded = new byte[large.length * 2];
        float[] decoded = new float[large.length];

        serializer.floatToByteArray(large, encoded, large.length);
        serializer.byteToFloatArray(encoded, decoded, large.length, 0);

        for (int i = 0; i < large.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(large[i]));
            assertEquals(expected, decoded[i], FP16_TOLERANCE);
        }
    }

    private float[] getArrayOfRandomFloats(int length) {
        float[] vector = new float[length];
        for (int i = 0; i < length; i++) {
            vector[i] = random.nextFloat() * 200 - 100; // Range: [-100, 100]
        }
        return vector;
    }
}
