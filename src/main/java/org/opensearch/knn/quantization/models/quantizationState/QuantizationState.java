/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

import java.io.IOException;
import java.io.Serializable;

/**
 * QuantizationState interface represents the state of a quantization process, including the parameters used.
 * This interface provides methods for serializing and deserializing the state.
 */
public interface QuantizationState extends Serializable {
    /**
     * Returns the quantization parameters associated with this state.
     *
     * @return the quantization parameters.
     */
    QuantizationParams getQuantizationParams();

    /**
     * Serializes the quantization state to a byte array.
     *
     * @return a byte array representing the serialized state.
     * @throws IOException if an I/O error occurs during serialization.
     */
    byte[] toByteArray() throws IOException;
}
