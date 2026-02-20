/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.Version;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import static org.opensearch.knn.common.KNNConstants.BYTE_ALIGNMENT_MASK;

import java.io.IOException;

/**
 * OneBitScalarQuantizationState represents the state of one-bit scalar quantization,
 * including the mean values used for quantization.
 */
@Getter
@NoArgsConstructor // No-argument constructor for deserialization
@AllArgsConstructor
public final class OneBitScalarQuantizationState implements QuantizationState {
    private ScalarQuantizationParams quantizationParams;
    /**
     * Mean thresholds used in the quantization process.
     * Each threshold value corresponds to a dimension of the vector being quantized.
     *
     * Example:
     * If we have a vector [1.2, 3.4, 5.6] and mean thresholds [2.0, 3.0, 4.0],
     * The quantized vector will be [0, 1, 1].
     */
    private float[] meanThresholds;

    @Override
    public ScalarQuantizationParams getQuantizationParams() {
        return quantizationParams;
    }

    /**
     * This method is responsible for writing the state of the OneBitScalarQuantizationState object to an external output.
     * It includes versioning information to ensure compatibility between different versions of the serialized object.
     * @param out the StreamOutput to write the object to.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(Version.CURRENT.id); // Write the version
        quantizationParams.writeTo(out);
        out.writeFloatArray(meanThresholds);
    }

    /**
     * This method is responsible for reading the state of the OneBitScalarQuantizationState object from an external input.
     * It includes versioning information to ensure compatibility between different versions of the serialized object.
     * @param in the StreamInput to read the object from.
     * @throws IOException if an I/O error occurs during deserialization.
     */
    public OneBitScalarQuantizationState(StreamInput in) throws IOException {
        int version = in.readVInt(); // Read the version
        this.quantizationParams = new ScalarQuantizationParams(in, version);
        this.meanThresholds = in.readFloatArray();
    }

    /**
     * Serializes the current state of this OneBitScalarQuantizationState object into a byte array.
     * This method uses the QuantizationStateSerializer to handle the serialization process.
     *
     * <p>The serialized byte array includes all necessary state information, such as the mean thresholds
     * and quantization parameters, ensuring that the object can be fully reconstructed from the byte array.</p>
     *
     * <pre>
     * {@code
     * OneBitScalarQuantizationState state = new OneBitScalarQuantizationState(params, meanThresholds);
     * byte[] serializedState = state.toByteArray();
     * }
     * </pre>
     *
     * @return a byte array representing the serialized state of this object.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public byte[] toByteArray() throws IOException {
        return QuantizationStateSerializer.serialize(this);
    }

    /**
     * Deserializes a OneBitScalarQuantizationState object from a byte array.
     * This method uses the QuantizationStateSerializer to handle the deserialization process.
     *
     * <p>The byte array should contain serialized state information, including the mean thresholds
     * and quantization parameters, which are necessary to reconstruct the OneBitScalarQuantizationState object.</p>
     *
     * <pre>
     * {@code
     * byte[] serializedState = ...; // obtain the byte array from some source
     * OneBitScalarQuantizationState state = OneBitScalarQuantizationState.fromByteArray(serializedState);
     * }
     * </pre>
     *
     * @param bytes the byte array containing the serialized state.
     * @return the deserialized OneBitScalarQuantizationState object.
     * @throws IOException if an I/O error occurs during deserialization.
     */
    public static OneBitScalarQuantizationState fromByteArray(final byte[] bytes) throws IOException {
        return (OneBitScalarQuantizationState) QuantizationStateSerializer.deserialize(bytes, OneBitScalarQuantizationState::new);
    }

    /**
     * Calculates and returns the number of bytes stored per vector after quantization.
     *
     * @return the number of bytes stored per vector.
     */
    @Override
    public int getBytesPerVector() {
        // Calculate the number of bytes required for one-bit quantization
        return (meanThresholds.length + BYTE_ALIGNMENT_MASK) / Byte.SIZE;
    }

    @Override
    public int getDimensions() {
        // For one-bit quantization, the dimension for indexing is just the length of the thresholds array.
        // Align the original dimensions to the next multiple of 8
        return (meanThresholds.length + BYTE_ALIGNMENT_MASK) & ~BYTE_ALIGNMENT_MASK;
    }

    /**
     * Calculates the memory usage of the OneBitScalarQuantizationState object in bytes.
     * This method computes the shallow size of the instance itself, the shallow size of the
     * quantization parameters, and the memory usage of the mean thresholds array.
     *
     * @return The estimated memory usage of the OneBitScalarQuantizationState object in bytes.
     */
    @Override
    public long ramBytesUsed() {
        long size = RamUsageEstimator.shallowSizeOfInstance(OneBitScalarQuantizationState.class);
        size += RamUsageEstimator.shallowSizeOf(quantizationParams);
        size += RamUsageEstimator.sizeOf(meanThresholds);
        return size;
    }
}
