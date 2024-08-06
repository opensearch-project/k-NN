/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.knn.quantization.util;

import lombok.experimental.UtilityClass;

import java.util.List;

/**
 * Utility class for bit packing operations.
 * Provides methods for packing arrays of bits into byte arrays for efficient storage or transmission.
 */
@UtilityClass
public class BitPacker {

    /**
     * Packs the list of bit arrays into a single byte array.
     * Each byte in the resulting array contains up to 8 bits from the bit arrays, packed from left to right.
     *
     * @param bitArrays the list of bit arrays to be packed. Each bit array should contain only 0s and 1s.
     * @return a byte array containing the packed bits.
     * @throws IllegalArgumentException if the bitArrays list is empty, if any bit array is null, or if bit arrays have inconsistent lengths.
     */
    public static byte[] packBits(List<byte[]> bitArrays) {
        if (bitArrays.isEmpty()) {
            throw new IllegalArgumentException("The list of bit arrays cannot be empty.");
        }

        int bitArrayLength = bitArrays.get(0).length;
        int bitLength = bitArrays.size() * bitArrayLength;
        int byteLength = (bitLength + 7) / 8;
        byte[] packedArray = new byte[byteLength];

        int bitPosition = 0;
        for (byte[] bitArray : bitArrays) {
            if (bitArray == null) {
                throw new IllegalArgumentException("Bit array cannot be null.");
            }
            if (bitArray.length != bitArrayLength) {
                throw new IllegalArgumentException("All bit arrays must have the same length.");
            }

            for (byte bit : bitArray) {
                int byteIndex = bitPosition / 8;
                int bitIndex = 7 - (bitPosition % 8);
                if (bit == 1) {
                    packedArray[byteIndex] |= (1 << bitIndex);
                }
                bitPosition++;
            }
        }

        return packedArray;
    }
}
