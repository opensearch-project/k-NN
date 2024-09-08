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
     * This method provides access to the quantized data in its current state.
     * It returns the same reference to the internal quantized vector on each call, meaning any modifications
     * to the returned array will directly affect the internal state of the object. This design is intentional
     * to avoid unnecessary copying of data and improve performance, especially in scenarios where frequent
     * access to the quantized vector is required.
     *
     * <p><b>Important:</b> As this method returns a direct reference to the internal array, care must be taken
     * when modifying the returned array. If the returned vector is altered, the changes will reflect in the
     * quantized vector managed by the object, which could lead to unintended side effects.</p>
     *
     * <p><b>Usage Example:</b></p>
     * <pre>
     * byte[] quantizedData = quantizationOutput.getQuantizedVector();
     * // Use or modify quantizedData, but be cautious that changes affect the internal state.
     * </pre>
     *
     * This method does not create a deep copy of the vector to avoid performance overhead in real-time
     * or high-frequency operations. If a separate copy of the vector is needed, the caller should manually
     * clone or copy the returned array.
     *
     * <p><b>Example to clone the array:</b></p>
     * <pre>
     * byte[] clonedData = Arrays.copyOf(quantizationOutput.getQuantizedVector(), quantizationOutput.getQuantizedVector().length);
     * </pre>
     *
     * @return the quantized vector (same reference on each invocation).
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
