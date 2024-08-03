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
}
