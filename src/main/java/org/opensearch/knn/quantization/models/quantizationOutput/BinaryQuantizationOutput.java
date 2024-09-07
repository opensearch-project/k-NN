/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationOutput;

import lombok.Getter;

import java.util.Arrays;
import lombok.RequiredArgsConstructor;

/**
 * The BinaryQuantizationOutput class represents the output of a quantization process in binary format.
 * It implements the QuantizationOutput interface to handle byte arrays specifically.
 */
@Getter
@RequiredArgsConstructor
public class BinaryQuantizationOutput implements QuantizationOutput<byte[]> {
    private byte[] quantizedVector;
    private final int bitsPerCoordinate;
    private int currentVectorLength = -1; // Indicates uninitialized state

    /**
     * Prepares the quantized vector based on the vector length.
     * This includes initializing or resetting the quantized vector.
     *
     * @param vectorLength The length of the vector to be quantized.
     */
    @Override
    public void prepareQuantizedVector(int vectorLength) {
        if (vectorLength <= 0) {
            throw new IllegalArgumentException("Vector length must be greater than zero.");
        }

        if (vectorLength != currentVectorLength) {
            int totalBits = bitsPerCoordinate * vectorLength;
            int byteLength = (totalBits + 7) >> 3;
            this.quantizedVector = new byte[byteLength];
            this.currentVectorLength = vectorLength;
        } else {
            Arrays.fill(this.quantizedVector, (byte) 0);
        }
    }

    /**
     * Checks if the quantized vector has already been prepared for the given vector length.
     *
     * @param vectorLength The length of the vector to be quantized.
     * @return true if the quantized vector is already prepared, false otherwise.
     */
    @Override
    public boolean isPrepared(int vectorLength) {
        return vectorLength == currentVectorLength && quantizedVector != null;
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

    /**
     * Returns a copy of the quantized vector.
     *
     * @return a copy of the quantized vector byte array.
     */
    @Override
    public byte[] getQuantizedVectorCopy() {
        byte[] clonedByteArray = new byte[quantizedVector.length];
        System.arraycopy(quantizedVector, 0, clonedByteArray, 0, quantizedVector.length);
        return clonedByteArray;
    }
}
