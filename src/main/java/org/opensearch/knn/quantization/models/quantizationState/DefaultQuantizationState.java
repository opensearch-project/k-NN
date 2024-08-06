/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.util.QuantizationStateSerializer;

import java.io.IOException;

/**
 * DefaultQuantizationState is used as a fallback state when no training is required or if training fails.
 * It can be utilized by any quantizer to represent a default state.
 */
public class DefaultQuantizationState implements QuantizationState {

    private final QuantizationParams params;

    /**
     * Constructs a DefaultQuantizationState with the given quantization parameters.
     *
     * @param params the quantization parameters.
     */
    public DefaultQuantizationState(final QuantizationParams params) {
        this.params = params;
    }

    /**
     * Returns the quantization parameters associated with this state.
     *
     * @return the quantization parameters.
     */
    @Override
    public QuantizationParams getQuantizationParams() {
        return params;
    }

    /**
     * Serializes the quantization state to a byte array.
     *
     * @return a byte array representing the serialized state.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public byte[] toByteArray() throws IOException {
        return QuantizationStateSerializer.serialize(this, null);
    }

    /**
     * Deserializes a DefaultQuantizationState from a byte array.
     *
     * @param bytes the byte array containing the serialized state.
     * @return the deserialized DefaultQuantizationState.
     * @throws IOException            if an I/O error occurs during deserialization.
     * @throws ClassNotFoundException if the class of the serialized object cannot be found.
     */
    public static DefaultQuantizationState fromByteArray(final byte[] bytes) throws IOException, ClassNotFoundException {
        return (DefaultQuantizationState) QuantizationStateSerializer.deserialize(
            bytes,
            (parentParams, specificData) -> new DefaultQuantizationState((SQParams) parentParams)
        );
    }
}
