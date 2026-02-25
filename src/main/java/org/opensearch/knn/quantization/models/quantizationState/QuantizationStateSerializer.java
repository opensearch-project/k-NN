/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.experimental.UtilityClass;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;

import java.io.IOException;

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
        QuantizationState deserialize(StreamInput in) throws IOException;
    }

    /**
     * Serializes the QuantizationState and specific data into a byte array.
     *
     * @param state         The QuantizationState to serialize.
     * @return A byte array representing the serialized state and specific data.
     * @throws IOException If an I/O error occurs during serialization.
     */
    static byte[] serialize(QuantizationState state) throws IOException {
        try (BytesStreamOutput out = new BytesStreamOutput()) {
            state.writeTo(out);
            return out.bytes().toBytesRef().bytes;
        }
    }

    /**
     * Deserializes a QuantizationState and its specific data from a byte array.
     *
     * @param bytes                    The byte array containing the serialized data.
     * @param deserializer             The deserializer for the specific data associated with the state.
     * @return The deserialized QuantizationState including its specific data.
     * @throws IOException            If an I/O error occurs during deserialization.
     */
    static QuantizationState deserialize(byte[] bytes, SerializableDeserializer deserializer) throws IOException {
        try (StreamInput in = StreamInput.wrap(bytes)) {
            return deserializer.deserialize(in);
        }
    }
}
