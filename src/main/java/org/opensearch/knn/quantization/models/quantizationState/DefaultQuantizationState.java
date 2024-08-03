/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.Version;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

/**
 * DefaultQuantizationState is used as a fallback state when no training is required or if training fails.
 * It can be utilized by any quantizer to represent a default state.
 */
@Getter
@NoArgsConstructor // No-argument constructor for deserialization
@AllArgsConstructor
public class DefaultQuantizationState implements QuantizationState {
    private QuantizationParams params;
    private static final long serialVersionUID = 1L; // Version ID for serialization

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
            new DefaultQuantizationState(),
            (parentParams, specificData) -> new DefaultQuantizationState((ScalarQuantizationParams) parentParams)
        );
    }

    /**
     * Writes the object to the output stream.
     * This method is part of the Externalizable interface and is used to serialize the object.
     *
     * @param out the output stream to write the object to.
     * @throws IOException if an I/O error occurs.
     */
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(Version.CURRENT.id); // Write the version
        out.writeObject(params);
    }

    /**
     * Reads the object from the input stream.
     * This method is part of the Externalizable interface and is used to deserialize the object.
     *
     * @param in the input stream to read the object from.
     * @throws IOException if an I/O error occurs.
     * @throws ClassNotFoundException if the class of the serialized object cannot be found.
     */
    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        this.params = (QuantizationParams) in.readObject();
    }
}
