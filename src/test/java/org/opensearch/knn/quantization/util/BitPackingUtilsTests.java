/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.util;

import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;
import java.util.List;

public class BitPackingUtilsTests extends KNNTestCase {

    public void testPackBits() {
        List<byte[]> bitArrays = Arrays.asList(new byte[] { 0, 1, 0, 1, 1, 0, 1, 1 }, new byte[] { 1, 0, 1, 0, 0, 1, 0, 0 });

        byte[] expectedPackedArray = new byte[] { (byte) 0b01011011, (byte) 0b10100100 };
        byte[] packedArray = BitPacker.packBits(bitArrays);

        assertArrayEquals(expectedPackedArray, packedArray);
    }

    public void testPackBitsEmptyList() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> { BitPacker.packBits(Arrays.asList()); });
        assertEquals("The list of bit arrays cannot be empty.", exception.getMessage());
    }

    public void testPackBitsNullBitArray() {
        List<byte[]> bitArrays = Arrays.asList(new byte[] { 0, 1, 0, 1, 1, 0, 1, 1 }, null);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> { BitPacker.packBits(bitArrays); });
        assertEquals("Bit array cannot be null.", exception.getMessage());
    }

    public void testPackBitsInconsistentLength() {
        List<byte[]> bitArrays = Arrays.asList(new byte[] { 0, 1, 0, 1, 1, 0, 1, 1 }, new byte[] { 1, 0, 1 });

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> { BitPacker.packBits(bitArrays); });
        assertEquals("All bit arrays must have the same length.", exception.getMessage());
    }

    public void testPackBitsEdgeCaseSingleBitArray() {
        List<byte[]> bitArrays = Arrays.asList(new byte[] { 1 });

        byte[] expectedPackedArray = new byte[] { (byte) 0b10000000 };
        byte[] packedArray = BitPacker.packBits(bitArrays);

        assertArrayEquals("Packed array does not match expected output.", expectedPackedArray, packedArray);
    }

    public void testPackBitsEdgeCaseSingleBit() {
        List<byte[]> bitArrays = Arrays.asList(new byte[] { 1, 0, 1, 0, 1, 0, 1, 0 }, new byte[] { 1, 1, 1, 1, 1, 1, 1, 1 });

        byte[] expectedPackedArray = new byte[] { (byte) 0b10101010, (byte) 0b11111111 };
        byte[] packedArray = BitPacker.packBits(bitArrays);

        assertArrayEquals("Packed array does not match expected output.", expectedPackedArray, packedArray);
    }

    public void testPackBits_emptyArray() {
        List<byte[]> bitArrays = Arrays.asList();
        expectThrows(IllegalArgumentException.class, () -> { BitPacker.packBits(bitArrays); });
        ;
    }
}
