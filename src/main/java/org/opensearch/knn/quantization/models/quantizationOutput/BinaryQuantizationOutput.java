/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationOutput;

import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.Arrays;

/**
 * The BinaryQuantizationOutput class represents the output of a quantization process in binary format.
 * It implements the QuantizationOutput interface to handle byte arrays specifically.
 */
@NoArgsConstructor
public class BinaryQuantizationOutput implements QuantizationOutput<byte[]> {
    @Getter
    private byte[] quantizedVector;

    /**
     * Prepares the quantized vector array based on the provided parameters and returns it for direct modification.
     * This method ensures that the internal byte array is appropriately sized and cleared before being used.
     * The method accepts two parameters:
     * <ul>
     *     <li><b>bitsPerCoordinate:</b> The number of bits used per coordinate. This determines the granularity of the quantization.</li>
     *     <li><b>vectorLength:</b> The length of the original vector that needs to be quantized. This helps in calculating the required byte array size.</li>
     * </ul>
     * If the existing quantized vector is either null or not the same size as the required byte array,
     * a new byte array is allocated. Otherwise, the existing array is cleared (i.e., all bytes are set to zero).
     * This method is designed to be used in conjunction with a bit-packing utility that writes quantized values directly
     * into the returned byte array.
     * @param params an array of parameters, where the first parameter is the number of bits per coordinate (int),
     *               and the second parameter is the length of the vector (int).
     * @return the prepared and writable quantized vector as a byte array.
     * @throws IllegalArgumentException if the parameters are not as expected (e.g., missing or not integers).
     */
    @Override
    public byte[] prepareAndGetWritableQuantizedVector(Object... params) {
        if (params.length != 2 || !(params[0] instanceof Integer) || !(params[1] instanceof Integer)) {
            throw new IllegalArgumentException("Expected two integer parameters: bitsPerCoordinate and vectorLength");
        }
        int bitsPerCoordinate = (int) params[0];
        int vectorLength = (int) params[1];
        int totalBits = bitsPerCoordinate * vectorLength;
        int byteLength = (totalBits + 7) >> 3;

        if (this.quantizedVector == null || this.quantizedVector.length != byteLength) {
            this.quantizedVector = new byte[byteLength];
        } else {
            Arrays.fill(this.quantizedVector, (byte) 0);
        }

        return this.quantizedVector;
    }

    /**
     * Returns the quantized vector.
     *
     * @return the quantized vector byte array.
     */
    @Override
    public byte[] getQuantizedVector() {
        return quantizedVector;
    }
}
