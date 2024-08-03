/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.experimental.UtilityClass;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.io.IOException;
import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;

/**
 * QuantizationStateSerializer is a utility class that provides methods for serializing and deserializing
 * QuantizationState objects along with their specific data.
 */
@UtilityClass
class QuantizationStateSerializer {

    /**
     * A functional interface for deserializing specific data associated with a QuantizationState.
     */
    @FunctionalInterface
    interface SerializableDeserializer {
        QuantizationState deserialize(QuantizationParams parentParams, Serializable specificData);
    }

    /**
     * Serializes the QuantizationState and specific data into a byte array.
     *
     * @param state         The QuantizationState to serialize.
     * @param specificData  The specific data related to the state, to be serialized.
     * @return A byte array representing the serialized state and specific data.
     * @throws IOException If an I/O error occurs during serialization.
     */
    static byte[] serialize(QuantizationState state, Serializable specificData) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(); ObjectOutputStream out = new ObjectOutputStream(bos)) {
            state.writeExternal(out);
            out.writeObject(specificData);
            out.flush();
            return bos.toByteArray();
        }
    }

    /**
     * Deserializes a QuantizationState and its specific data from a byte array.
     *
     * @param bytes                    The byte array containing the serialized data.
     * @param stateInstance            An instance of the state to call readExternal on.
     * @param specificDataDeserializer The deserializer for the specific data associated with the state.
     * @return The deserialized QuantizationState including its specific data.
     * @throws IOException            If an I/O error occurs during deserialization.
     * @throws ClassNotFoundException If the class of the serialized object cannot be found.
     */
    static QuantizationState deserialize(byte[] bytes, QuantizationState stateInstance, SerializableDeserializer specificDataDeserializer)
        throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes); ObjectInputStream in = new ObjectInputStream(bis)) {
            stateInstance.readExternal(in);
            Serializable specificData = (Serializable) in.readObject(); // Read the specific data
            return specificDataDeserializer.deserialize(stateInstance.getQuantizationParams(), specificData);
        }
    }
}
