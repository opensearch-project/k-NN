/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.util;

import lombok.experimental.UtilityClass;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

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
public class QuantizationStateSerializer {

    /**
     * A functional interface for deserializing specific data associated with a QuantizationState.
     */
    @FunctionalInterface
    public interface SerializableDeserializer {
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
    public static byte[] serialize(QuantizationState state, Serializable specificData) throws IOException {
        byte[] parentBytes = serializeParentParams(state.getQuantizationParams());
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(); ObjectOutputStream out = new ObjectOutputStream(bos)) {
            out.writeInt(parentBytes.length); // Write the length of the parent bytes
            out.write(parentBytes); // Write the parent bytes
            out.writeObject(specificData); // Write the specific data
            out.flush();
            return bos.toByteArray();
        }
    }

    /**
     * Deserializes a QuantizationState and its specific data from a byte array.
     *
     * @param bytes                    The byte array containing the serialized data.
     * @param specificDataDeserializer The deserializer for the specific data associated with the state.
     * @return The deserialized QuantizationState including its specific data.
     * @throws IOException            If an I/O error occurs during deserialization.
     * @throws ClassNotFoundException If the class of the serialized object cannot be found.
     */
    public static QuantizationState deserialize(byte[] bytes, SerializableDeserializer specificDataDeserializer) throws IOException,
        ClassNotFoundException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes); ObjectInputStream in = new ObjectInputStream(bis)) {
            int parentLength = in.readInt();
            // Read the length of the parent bytes
            byte[] parentBytes = new byte[parentLength];
            in.readFully(parentBytes); // Read the parent bytes
            QuantizationParams parentParams = deserializeParentParams(parentBytes); // Deserialize the parent params
            Serializable specificData = (Serializable) in.readObject(); // Read the specific data
            return specificDataDeserializer.deserialize(parentParams, specificData);
        }
    }

    /**
     * Serializes the parent parameters of the QuantizationState into a byte array.
     *
     * @param params The QuantizationParams to serialize.
     * @return A byte array representing the serialized parent parameters.
     * @throws IOException If an I/O error occurs during serialization.
     */
    private static byte[] serializeParentParams(QuantizationParams params) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(); ObjectOutputStream out = new ObjectOutputStream(bos)) {
            out.writeObject(params);
            out.flush();
            return bos.toByteArray();
        }
    }

    /**
     * Deserializes the parent parameters of the QuantizationState from a byte array.
     *
     * @param bytes The byte array containing the serialized parent parameters.
     * @return The deserialized QuantizationParams.
     * @throws IOException            If an I/O error occurs during deserialization.
     * @throws ClassNotFoundException If the class of the serialized object cannot be found.
     */
    private static QuantizationParams deserializeParentParams(byte[] bytes) throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes); ObjectInputStream in = new ObjectInputStream(bis)) {
            return (QuantizationParams) in.readObject();
        }
    }
}
