/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Builder;
import lombok.NonNull;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.Version;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;

import java.io.IOException;

/**
 * MultiBitScalarQuantizationState represents the state of multi-bit scalar quantization,
 * including the thresholds used for quantization.
 */
@Getter
@Builder
@AllArgsConstructor
@NoArgsConstructor(force = true)
public final class MultiBitScalarQuantizationState implements QuantizationState {

    @NonNull
    private ScalarQuantizationParams quantizationParams;
    /**
     * The threshold values for multi-bit quantization, organized as a 2D array
     * where each row corresponds to a different bit level.
     *
     * For example:
     * - For 2-bit quantization:
     *   thresholds[0] {0.5f, 1.5f, 2.5f}  // Thresholds for the first bit level
     *   thresholds[1] {1.0f, 2.0f, 3.0f}  // Thresholds for the second bit level
     * - For 4-bit quantization:
     *   thresholds[0] {0.1f, 0.2f, 0.3f}
     *   thresholds[1] {0.4f, 0.5f, 0.6f}
     *   thresholds[2] {0.7f, 0.8f, 0.9f}
     *   thresholds[3] {1.0f, 1.1f, 1.2f}
     *
     * Each column represents the threshold for a specific dimension in the vector space.
     */
    @NonNull
    private float[][] thresholds;

    @Builder.Default
    private float[] aboveThresholdMeans = null;

    @Builder.Default
    private float[] belowThresholdMeans = null;

    @Builder.Default
    private double averageL2L1Ratio = 0.0;

    @Builder.Default
    private float[][] rotationMatrix = null;

    @Override
    public ScalarQuantizationParams getQuantizationParams() {
        return quantizationParams;
    }

    /**
     * This method is responsible for writing the state of the MultiBitScalarQuantizationState object to an external output.
     * It includes versioning information to ensure compatibility between different versions of the serialized object.
     *
     * @param out the StreamOutput to write the object to.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(Version.CURRENT.id); // Write the version
        quantizationParams.writeTo(out);
        out.writeVInt(thresholds.length); // Number of rows
        for (float[] row : thresholds) {
            out.writeFloatArray(row); // Write each row as a float array
        }

        if (Version.CURRENT.onOrAfter(Version.V_3_0_0)) {
            out.writeOptionalArray(belowThresholdMeans != null ? new FloatArrayWrapper[]{new FloatArrayWrapper(belowThresholdMeans)} : null);
            out.writeOptionalArray(aboveThresholdMeans != null ? new FloatArrayWrapper[]{new FloatArrayWrapper(aboveThresholdMeans)} : null);
            out.writeOptionalDouble(averageL2L1Ratio);
            if (rotationMatrix != null) {
                out.writeBoolean(true);
                out.writeVInt(rotationMatrix.length);
                for (float[] row : rotationMatrix) {
                    out.writeFloatArray(row);
                }
            } else {
                out.writeBoolean(false);
            }
        }

    }

    /**
     * This method is responsible for reading the state of the MultiBitScalarQuantizationState object from an external input.
     * It includes versioning information to ensure compatibility between different versions of the serialized object.
     *
     * @param in the StreamInput to read the object from.
     * @throws IOException if an I/O error occurs during deserialization.
     */
    public MultiBitScalarQuantizationState(StreamInput in) throws IOException {
        int version = in.readVInt();
        this.quantizationParams = new ScalarQuantizationParams(in, version);

        int rows = in.readVInt();
        this.thresholds = new float[rows][];
        for (int i = 0; i < rows; i++) {
            this.thresholds[i] = in.readFloatArray();
        }

        if (Version.fromId(version).onOrAfter(Version.V_3_0_0)) {
            FloatArrayWrapper[] wrappedBelowMeans = in.readOptionalArray(FloatArrayWrapper::new, FloatArrayWrapper[]::new);
            this.belowThresholdMeans = wrappedBelowMeans != null ? wrappedBelowMeans[0].getArray() : null;

            FloatArrayWrapper[] wrappedAboveMeans = in.readOptionalArray(FloatArrayWrapper::new, FloatArrayWrapper[]::new);
            this.aboveThresholdMeans = wrappedAboveMeans != null ? wrappedAboveMeans[0].getArray() : null;

            this.averageL2L1Ratio = in.readOptionalDouble();

            if (in.readBoolean()) {
                int dims = in.readVInt();
                this.rotationMatrix = new float[dims][];
                for (int i = 0; i < dims; i++) {
                    this.rotationMatrix[i] = in.readFloatArray();
                }
            }
        }
    }

    /**
     * Serializes the current state of this MultiBitScalarQuantizationState object into a byte array.
     * This method uses the QuantizationStateSerializer to handle the serialization process.
     *
     * <p>The serialized byte array includes all necessary state information, such as the thresholds
     * and quantization parameters, ensuring that the object can be fully reconstructed from the byte array.</p>
     *
     * <pre>
     * {@code
     * MultiBitScalarQuantizationState state = new MultiBitScalarQuantizationState(params, thresholds);
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
     * Deserializes a MultiBitScalarQuantizationState object from a byte array.
     * This method uses the QuantizationStateSerializer to handle the deserialization process.
     *
     * <p>The byte array should contain serialized state information, including the thresholds
     * and quantization parameters, which are necessary to reconstruct the MultiBitScalarQuantizationState object.</p>
     *
     * <pre>
     * {@code
     * byte[] serializedState = ...; // obtain the byte array from some source
     * MultiBitScalarQuantizationState state = MultiBitScalarQuantizationState.fromByteArray(serializedState);
     * }
     * </pre>
     *
     * @param bytes the byte array containing the serialized state.
     * @return the deserialized MultiBitScalarQuantizationState object.
     * @throws IOException if an I/O error occurs during deserialization.
     */
    public static MultiBitScalarQuantizationState fromByteArray(final byte[] bytes) throws IOException {
        return (MultiBitScalarQuantizationState) QuantizationStateSerializer.deserialize(bytes, MultiBitScalarQuantizationState::new);
    }

    /**
     * Calculates and returns the number of bytes stored per vector after quantization.
     *
     * @return the number of bytes stored per vector.
     */
    @Override
    public int getBytesPerVector() {
        // Check if thresholds are null or have invalid structure
        if (thresholds == null || thresholds.length == 0 || thresholds[0] == null) {
            throw new IllegalStateException("Error in getBytesStoredPerVector: The thresholds array is not initialized.");
        }

        // Calculate the number of bytes required for multi-bit quantization
        return thresholds.length * thresholds[0].length;
    }

    @Override
    public int getDimensions() {
        // For multi-bit quantization, the dimension for indexing is the number of rows * columns in the thresholds array.
        // Where number of column reprensents Dimesion of Original vector and number of rows equals to number of bits
        // Check if thresholds are null or have invalid structure
        if (thresholds == null || thresholds.length == 0 || thresholds[0] == null) {
            throw new IllegalStateException("Error in getting Dimension: The thresholds array is not initialized.");
        }
        int originalDimensions = thresholds[0].length;

        // Align the original dimensions to the next multiple of 8 for each bit level
        int alignedDimensions = (originalDimensions + 7) & ~7;

        // The final dimension count should consider the bit levels
        return thresholds.length * alignedDimensions;

    }

    /**
     * Calculates the memory usage of the MultiBitScalarQuantizationState object in bytes.
     * This method computes the shallow size of the instance itself, the shallow size of the
     * quantization parameters, and the memory usage of the 2D thresholds array.
     *
     * @return The estimated memory usage of the MultiBitScalarQuantizationState object in bytes.
     */
    @Override
    public long ramBytesUsed() {
        long size = RamUsageEstimator.shallowSizeOfInstance(MultiBitScalarQuantizationState.class);
        size += RamUsageEstimator.shallowSizeOf(quantizationParams);
        size += RamUsageEstimator.shallowSizeOf(thresholds); // shallow size of the 2D array (array of references to rows)
        for (float[] row : thresholds) {
            size += RamUsageEstimator.sizeOf(row); // size of each row in the 2D array
        }
        if (belowThresholdMeans != null) {
            size += RamUsageEstimator.sizeOf(belowThresholdMeans);
        }
        if (aboveThresholdMeans != null) {
            size += RamUsageEstimator.sizeOf(aboveThresholdMeans);
        }
        if (rotationMatrix != null) {
            size += RamUsageEstimator.shallowSizeOf(rotationMatrix);
            for (float[] row : rotationMatrix) {
                size += RamUsageEstimator.sizeOf(row);
            }
        }
        return size;
    }

    private static class FloatArrayWrapper implements Writeable {
        private final float[] array;

        public FloatArrayWrapper(float[] array) {
            this.array = array;
        }

        public FloatArrayWrapper(StreamInput in) throws IOException {
            this.array = in.readFloatArray();
        }

        public float[] getArray() {
            return array;
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeFloatArray(array);
        }
    }
}
