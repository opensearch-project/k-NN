/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationOutput;

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
     * Prepares the quantized vector based on the vector length.
     * This includes initializing or resetting the quantized vector.
     *
     * @param vectorLength The length of the vector to be quantized.
     */
    void prepareQuantizedVector(int vectorLength);

    /**
     * Checks if the quantized vector has already been prepared for the given vector length.
     *
     * @param vectorLength The length of the vector to be quantized.
     * @return true if the quantized vector is already prepared, false otherwise.
     */
    boolean isPrepared(int vectorLength);

    /**
     * Returns a copy of the quantized vector.
     *
     * @return a copy of the quantized data.
     */
    T getQuantizedVectorCopy();
}
