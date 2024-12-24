/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationOutput;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public class ByteQuantizationOutput implements QuantizationOutput<byte[]> {
    private byte[] quantizedVector;
    private final int bitsPerCoordinate;
    private int currentVectorLength = -1; // Indicates uninitialized state

    @Override
    public byte[] getQuantizedVector() {
        return quantizedVector;
    }

    @Override
    public void prepareQuantizedVector(int vectorLength) {
        if (vectorLength <= 0) {
            throw new IllegalArgumentException("Vector length must be greater than zero.");
        }
        this.quantizedVector = new byte[vectorLength];
        this.currentVectorLength = vectorLength;

    }

    @Override
    public boolean isPrepared(int vectorLength) {
        return vectorLength == currentVectorLength;
    }

    @Override
    public byte[] getQuantizedVectorCopy() {
        byte[] clonedByteArray = new byte[quantizedVector.length];
        System.arraycopy(quantizedVector, 0, clonedByteArray, 0, quantizedVector.length);
        return clonedByteArray;
    }
}
