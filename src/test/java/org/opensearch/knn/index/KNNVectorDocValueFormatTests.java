/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class KNNVectorDocValueFormatTests extends KNNTestCase {

    public void testIsBinary() {
        assertFalse("ARRAY_FORMAT should not be binary", KNNVectorDocValueFormat.ARRAY_FORMAT.isBinary());
        assertTrue("BINARY_FORMAT should be binary", KNNVectorDocValueFormat.BINARY_FORMAT.isBinary());
    }

    public void testGetWriteableName() {
        assertEquals("ARRAY_FORMAT writeable name mismatch", "knn_vector", KNNVectorDocValueFormat.ARRAY_FORMAT.getWriteableName());
        assertEquals("BINARY_FORMAT writeable name mismatch", "knn_vector", KNNVectorDocValueFormat.BINARY_FORMAT.getWriteableName());
    }

    public void testStreamRoundTrip() throws IOException {
        // Array format round-trip
        BytesStreamOutput arrayOut = new BytesStreamOutput();
        KNNVectorDocValueFormat.ARRAY_FORMAT.writeTo(arrayOut);
        try (StreamInput arrayIn = arrayOut.bytes().streamInput()) {
            KNNVectorDocValueFormat arrayDeserialized = KNNVectorDocValueFormat.fromStream(arrayIn);
            assertFalse("Deserialized ARRAY_FORMAT should not be binary", arrayDeserialized.isBinary());
            assertEquals("Deserialized ARRAY_FORMAT writeable name mismatch", "knn_vector", arrayDeserialized.getWriteableName());
            assertSame(
                "Deserialized ARRAY_FORMAT should be the singleton instance",
                KNNVectorDocValueFormat.ARRAY_FORMAT,
                arrayDeserialized
            );
        }

        // Binary format round-trip
        BytesStreamOutput binaryOut = new BytesStreamOutput();
        KNNVectorDocValueFormat.BINARY_FORMAT.writeTo(binaryOut);
        try (StreamInput binaryIn = binaryOut.bytes().streamInput()) {
            KNNVectorDocValueFormat binaryDeserialized = KNNVectorDocValueFormat.fromStream(binaryIn);
            assertTrue("Deserialized BINARY_FORMAT should be binary", binaryDeserialized.isBinary());
            assertEquals("Deserialized BINARY_FORMAT writeable name mismatch", "knn_vector", binaryDeserialized.getWriteableName());
            assertSame(
                "Deserialized BINARY_FORMAT should be the singleton instance",
                KNNVectorDocValueFormat.BINARY_FORMAT,
                binaryDeserialized
            );
        }
    }

    public void testFromFormatString() {
        // null defaults to binary
        assertSame(
            "null format should return BINARY_FORMAT",
            KNNVectorDocValueFormat.BINARY_FORMAT,
            KNNVectorDocValueFormat.fromFormatString(null)
        );

        // explicit "binary" returns BINARY_FORMAT
        assertSame(
            "'binary' should return BINARY_FORMAT",
            KNNVectorDocValueFormat.BINARY_FORMAT,
            KNNVectorDocValueFormat.fromFormatString("binary")
        );

        // explicit "array" returns ARRAY_FORMAT
        assertSame(
            "'array' should return ARRAY_FORMAT",
            KNNVectorDocValueFormat.ARRAY_FORMAT,
            KNNVectorDocValueFormat.fromFormatString("array")
        );

        // unsupported format throws
        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorDocValueFormat.fromFormatString("epoch_millis")
        );
        assertTrue("Error should mention unsupported format", ex.getMessage().contains("epoch_millis"));
        assertTrue("Error should list supported formats", ex.getMessage().contains("array"));
        assertTrue("Error should list supported formats", ex.getMessage().contains("binary"));
    }

    public void testFloatToLittleEndianBytes() {
        // Simple vector
        float[] vector = { 1.0f, 2.0f, 3.0f, 4.0f };
        byte[] bytes = KNNVectorDocValueFormat.floatToLittleEndianBytes(vector);
        assertNotNull("Byte array should not be null", bytes);
        assertEquals("Byte array length should be dimension * 4", vector.length * Float.BYTES, bytes.length);
        assertDecodedVectorEquals("Simple vector", vector, bytes);

        // Single element
        float[] single = { 42.5f };
        assertDecodedVectorEquals("Single element vector", single, KNNVectorDocValueFormat.floatToLittleEndianBytes(single));

        // Negative and edge values
        float[] edgeCases = { -1.5f, -100.0f, 0.0f, Float.MAX_VALUE, Float.MIN_VALUE };
        assertDecodedVectorEquals("Edge case vector", edgeCases, KNNVectorDocValueFormat.floatToLittleEndianBytes(edgeCases));

        // Empty vector
        float[] empty = {};
        byte[] emptyBytes = KNNVectorDocValueFormat.floatToLittleEndianBytes(empty);
        assertNotNull("Empty vector encoding should not be null", emptyBytes);
        assertEquals("Empty vector should produce 0 bytes", 0, emptyBytes.length);

        // High dimension (768d)
        int dimension = 768;
        float[] highDim = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            highDim[i] = i * 0.01f;
        }
        assertDecodedVectorEquals("768d vector", highDim, KNNVectorDocValueFormat.floatToLittleEndianBytes(highDim));
    }

    public void testFloatToLittleEndianBytes_nullVector_throwsNPE() {
        expectThrows(NullPointerException.class, () -> KNNVectorDocValueFormat.floatToLittleEndianBytes(null));
    }

    public void testFloatToLittleEndianBytes_usesLittleEndian() {
        float[] vector = { 1.0f };
        byte[] bytes = KNNVectorDocValueFormat.floatToLittleEndianBytes(vector);

        // 1.0f in IEEE 754 is 0x3F800000
        // Little-endian: 0x00, 0x00, 0x80, 0x3F
        assertEquals("Byte 0 should be 0x00 (little-endian)", (byte) 0x00, bytes[0]);
        assertEquals("Byte 1 should be 0x00 (little-endian)", (byte) 0x00, bytes[1]);
        assertEquals("Byte 2 should be 0x80 (little-endian)", (byte) 0x80, bytes[2]);
        assertEquals("Byte 3 should be 0x3F (little-endian)", (byte) 0x3F, bytes[3]);
    }

    public void testToString() {
        assertEquals("ARRAY_FORMAT toString mismatch", "knn_vector(array)", KNNVectorDocValueFormat.ARRAY_FORMAT.toString());
        assertEquals("BINARY_FORMAT toString mismatch", "knn_vector(binary)", KNNVectorDocValueFormat.BINARY_FORMAT.toString());
    }

    private void assertDecodedVectorEquals(String label, float[] expected, byte[] bytes) {
        assertEquals(label + ": byte length mismatch", expected.length * Float.BYTES, bytes.length);

        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(label + ": value mismatch at index " + i, expected[i], buffer.getFloat(), 0.0f);
        }
    }
}
