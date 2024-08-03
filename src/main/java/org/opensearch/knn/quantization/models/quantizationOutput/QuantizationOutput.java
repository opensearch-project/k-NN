/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationOutput;

import java.io.IOException;

/**
 * The QuantizationOutput interface defines the contract for quantization output data.
 *
 * @param <T> The type of the quantized data.
 */
public interface QuantizationOutput<T> {
    /**
     * Returns the quantized vector.
     *
     * @return the quantized data.
     */
    T getQuantizedVector();

    /**
     * Updates the quantized vector with new data.
     *
     * @param newQuantizedVector the new quantized vector data.
     * @throws IOException if an I/O error occurs during the update.
     */
    void updateQuantizedVector(T newQuantizedVector) throws IOException;
}
