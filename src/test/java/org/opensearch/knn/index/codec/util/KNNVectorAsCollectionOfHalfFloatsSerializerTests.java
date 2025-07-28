/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.KNNTestCase;

import java.util.Random;
import java.util.stream.IntStream;

public class KNNVectorAsCollectionOfHalfFloatsSerializerTests extends KNNTestCase {

    Random random = new Random();
    private static final float FP16_TOLERANCE = 1e-3f;

    public void testVectorAsCollectionOfHalfFloatsSerializer() {
        float[] original = getArrayOfRandomFloats(20);
        KNNVectorSerializer serializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(original.length);

        // Serialize
        byte[] encoded = serializer.floatToByteArray(original);
        assertNotNull(encoded);
        assertEquals(original.length * 2, encoded.length, FP16_TOLERANCE);

        // Deserialize
        float[] decoded = serializer.byteToFloatArray(new BytesRef(encoded));
        assertNotNull(decoded);
        assertEquals(original.length, decoded.length, FP16_TOLERANCE);

        // Compare with fp16 precision tolerance
        for (int i = 0; i < original.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(original[i]));
            assertEquals(expected, decoded[i], FP16_TOLERANCE);
        }
    }

    public void testVectorSerializer_whenVectorBytesOffset_thenSuccess() {
        float[] original = getArrayOfRandomFloats(20);
        KNNVectorSerializer serializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(original.length);

        byte[] encoded = serializer.floatToByteArray(original);
        int offset = randomInt(4);

        byte[] padded = new byte[encoded.length + 2 * offset];
        System.arraycopy(encoded, 0, padded, offset, encoded.length);

        BytesRef refWithOffset = new BytesRef(padded, offset, encoded.length);
        float[] decoded = serializer.byteToFloatArray(refWithOffset);

        assertNotNull(decoded);
        assertEquals(original.length, decoded.length);
        for (int i = 0; i < original.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(original[i]));
            assertEquals(expected, decoded[i], FP16_TOLERANCE);
        }
    }

    public void testEmptyVector() {
        float[] empty = new float[0];
        KNNVectorSerializer serializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(0);
        byte[] encoded = serializer.floatToByteArray(empty);
        assertEquals(0, encoded.length);
        float[] decoded = serializer.byteToFloatArray(new BytesRef(encoded));
        assertEquals(0, decoded.length);
    }

    public void testInvalidByteStream_throwsException() {
        KNNVectorSerializer serializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(0);
        assertThrows(IllegalArgumentException.class, () -> serializer.byteToFloatArray(null));
        assertThrows(IllegalArgumentException.class, () -> serializer.byteToFloatArray(new BytesRef(new byte[3])));
        assertThrows(IllegalArgumentException.class, () -> serializer.byteToFloatArray(new BytesRef(new byte[1])));
    }

    public void testSpecialFloatValues() {
        float[] specialValues = new float[] { Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.0f, 0.0f };

        KNNVectorSerializer serializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(specialValues.length);
        byte[] encoded = serializer.floatToByteArray(specialValues);
        float[] decoded = serializer.byteToFloatArray(new BytesRef(encoded));

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

        KNNVectorSerializer serializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(boundaryValues.length);
        byte[] encoded = serializer.floatToByteArray(boundaryValues);
        float[] decoded = serializer.byteToFloatArray(new BytesRef(encoded));

        for (int i = 0; i < boundaryValues.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(boundaryValues[i]));
            assertEquals(expected, decoded[i], 0.0f);
        }
    }

    public void testLargeVector() {
        float[] large = getArrayOfRandomFloats(10000);
        KNNVectorSerializer serializer = new KNNVectorAsCollectionOfHalfFloatsSerializer(large.length);

        byte[] encoded = serializer.floatToByteArray(large);
        assertEquals(large.length * 2, encoded.length);

        float[] decoded = serializer.byteToFloatArray(new BytesRef(encoded));
        assertEquals(large.length, decoded.length);

        for (int i = 0; i < large.length; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(large[i]));
            assertEquals(expected, decoded[i], FP16_TOLERANCE);
        }
    }

    private float[] getArrayOfRandomFloats(int length) {
        float[] vector = new float[length];
        // Use fp16 safe range to avoid overflow
        IntStream.range(0, length).forEach(i -> vector[i] = random.nextFloat() * 200 - 100); // Range: [-100, 100]
        return vector;
    }
}
