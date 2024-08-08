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
     * Prepares and returns the writable quantized vector for direct modification.
     *
     * @param params the parameters needed for preparing the quantized vector.
     * @return the prepared and writable quantized vector.
     */
    T prepareAndGetWritableQuantizedVector(Object... params);
}
