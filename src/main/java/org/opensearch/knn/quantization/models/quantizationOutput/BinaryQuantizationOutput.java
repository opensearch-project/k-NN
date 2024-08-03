/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationOutput;

/**
 * The BinaryQuantizationOutput class represents the output of a quantization process in binary format.
 * It implements the QuantizationOutput interface to handle byte arrays specifically.
 */
public class BinaryQuantizationOutput implements QuantizationOutput<byte[]> {
    private final byte[] quantizedVector;

    /**
     * Constructs a BinaryQuantizationOutput instance with the specified quantized vector.
     *
     * @param quantizedVector the quantized vector represented as a byte array.
     */
    public BinaryQuantizationOutput(final byte[] quantizedVector) {
        if (quantizedVector == null) {
            throw new IllegalArgumentException("Quantized vector cannot be null");
        }
        this.quantizedVector = quantizedVector;
    }

    @Override
    public byte[] getQuantizedVector() {
        return quantizedVector;
    }
}
