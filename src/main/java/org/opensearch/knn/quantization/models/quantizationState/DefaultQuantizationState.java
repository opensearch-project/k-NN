/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.Version;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;

import java.io.IOException;

/**
 * DefaultQuantizationState is used as a fallback state when no training is required or if training fails.
 * It can be utilized by any quantizer to represent a default state.
 */
@Getter
@NoArgsConstructor // No-argument constructor for deserialization
@AllArgsConstructor
public class DefaultQuantizationState implements QuantizationState {
    private QuantizationParams params;

    @Override
    public QuantizationParams getQuantizationParams() {
        return params;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeInt(Version.CURRENT.id); // Write the version
        params.writeTo(out);
    }

    public DefaultQuantizationState(StreamInput in) throws IOException {
        int version = in.readInt(); // Read the version
        this.params = new ScalarQuantizationParams(in, version);
    }

    /**
     * Serializes the quantization state to a byte array.
     *
     * @return a byte array representing the serialized state.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public byte[] toByteArray() throws IOException {
        return QuantizationStateSerializer.serialize(this);
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
        return (DefaultQuantizationState) QuantizationStateSerializer.deserialize(bytes, DefaultQuantizationState::new);
    }

    @Override
    public int getBytesPerVector() {
        return 0;
    }

    @Override
    public int getDimensions() {
        return 0;
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }
}
