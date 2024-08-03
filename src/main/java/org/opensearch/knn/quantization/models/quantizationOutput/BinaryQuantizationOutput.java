/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationOutput;

import lombok.NoArgsConstructor;
import lombok.Getter;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * The BinaryQuantizationOutput class represents the output of a quantization process in binary format.
 * It implements the QuantizationOutput interface to handle byte arrays specifically.
 */
@NoArgsConstructor
public class BinaryQuantizationOutput implements QuantizationOutput<byte[]> {
    @Getter
    private final ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

    /**
     * Updates the quantized vector with a new byte array.
     *
     * @param newQuantizedVector the new quantized vector represented as a byte array.
     */
    public void updateQuantizedVector(final byte[] newQuantizedVector) throws IOException {
        if (newQuantizedVector == null || newQuantizedVector.length == 0) {
            throw new IllegalArgumentException("Quantized vector cannot be null or empty");
        }
        byteArrayOutputStream.reset();
        byteArrayOutputStream.write(newQuantizedVector);
    }

    @Override
    public byte[] getQuantizedVector() {
        return byteArrayOutputStream.toByteArray();
    }
}
