/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.quantization.models.quantizationOutput;

public abstract class QuantizationOutput<T> {
    private final T quantizedVector;

    public QuantizationOutput(T quantizedVector) {
        this.quantizedVector = quantizedVector;
    }

    public T getQuantizedVector() {
        return quantizedVector;
    }
}
