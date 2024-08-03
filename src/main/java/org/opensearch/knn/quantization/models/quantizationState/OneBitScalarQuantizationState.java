/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.Version;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.util.VersionContext;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

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
     * the quantization process will be:
     * - 1.2 < 2.0, so the first bit is 0
     * - 3.4 > 3.0, so the second bit is 1
     * - 5.6 > 4.0, so the third bit is 1
     * The quantized vector will be [0, 1, 1].
     */
    private float[] meanThresholds;
    private static final long serialVersionUID = 1L; // Version ID for serialization

    @Override
    public ScalarQuantizationParams getQuantizationParams() {
        return quantizationParams;
    }

    /**
     * This method is responsible for writing the state of the OneBitScalarQuantizationState object to an external output.
     * It includes versioning information to ensure compatibility between different versions of the serialized object.
     *
     * <p>Versioning is managed using the {@link VersionContext} class. This allows other classes that are serialized
     * as part of the state to access the version information and implement version-specific logic if needed.</p>
     *
     * <p>The {@link VersionContext#setVersion(int)} method sets the version information in a thread-local variable,
     * ensuring that the version is available to all classes involved in the serialization process within the current thread context.</p>
     *
     * <pre>
     * {@code
     * // Example usage in the writeExternal method:
     * VersionContext.setVersion(version);
     * out.writeInt(version); // Write the version
     * quantizationParams.writeExternal(out);
     * out.writeInt(meanThresholds.length);
     * for (float mean : meanThresholds) {
     *     out.writeFloat(mean);
     * }
     * }
     * </pre>
     *
     * @param out the ObjectOutput to write the object to.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        int version = Version.CURRENT.id;
        VersionContext.setVersion(version);
        out.writeInt(version); // Write the version
        quantizationParams.writeExternal(out);
        out.writeInt(meanThresholds.length);
        for (float mean : meanThresholds) {
            out.writeFloat(mean);
        }
        VersionContext.clear(); // Clear the version after use
    }

    /**
     * This method is responsible for reading the state of the OneBitScalarQuantizationState object from an external input.
     * It includes versioning information to ensure compatibility between different versions of the serialized object.
     *
     * <p>The version information is read first, and then it is set using the {@link VersionContext#setVersion(int)} method.
     * This makes the version information available to all classes involved in the deserialization process within the current thread context.</p>
     *
     * <p>Classes that are part of the deserialization process can retrieve the version information using the
     * {@link VersionContext#getVersion()} method and implement version-specific logic accordingly.</p>
     *
     * <pre>
     * {@code
     * // Example usage in the readExternal method:
     * int version = in.readInt(); // Read the version
     * VersionContext.setVersion(version);
     * quantizationParams = new ScalarQuantizationParams();
     * quantizationParams.readExternal(in); // Use readExternal of SQParams
     * int length = in.readInt();
     * meanThresholds = new float[length];
     * for (int i = 0; i < length; i++) {
     *     meanThresholds[i] = in.readFloat();
     * }
     * }
     * </pre>
     *
     * @param in the ObjectInput to read the object from.
     * @throws IOException if an I/O error occurs during deserialization.
     * @throws ClassNotFoundException if the class of the serialized object cannot be found.
     */
    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        int version = in.readInt(); // Read the version
        VersionContext.setVersion(version);
        quantizationParams = new ScalarQuantizationParams();
        quantizationParams.readExternal(in); // Use readExternal of SQParams
        int length = in.readInt();
        meanThresholds = new float[length];
        for (int i = 0; i < length; i++) {
            meanThresholds[i] = in.readFloat();
        }
        VersionContext.clear(); // Clear the version after use
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
        return QuantizationStateSerializer.serialize(this, meanThresholds);
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
     * @throws ClassNotFoundException if the class of a serialized object cannot be found.
     */
    public static OneBitScalarQuantizationState fromByteArray(final byte[] bytes) throws IOException, ClassNotFoundException {
        return (OneBitScalarQuantizationState) QuantizationStateSerializer.deserialize(
            bytes,
            new OneBitScalarQuantizationState(),
            (parentParams, meanThresholds) -> new OneBitScalarQuantizationState(
                (ScalarQuantizationParams) parentParams,
                (float[]) meanThresholds
            )
        );
    }
}
