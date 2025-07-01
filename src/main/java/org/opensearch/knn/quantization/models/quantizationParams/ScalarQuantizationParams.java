/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationParams;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.Version;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;

/**
 * The ScalarQuantizationParams class represents the parameters specific to scalar quantization (SQ).
 * This class implements the QuantizationParams interface and includes the type of scalar quantization.
 */
@Getter
@EqualsAndHashCode
@AllArgsConstructor
@NoArgsConstructor(force = true)
@Builder
public class ScalarQuantizationParams implements QuantizationParams {
    private final ScalarQuantizationType sqType;
    @Builder.Default
    private final boolean enableRandomRotation = QFrameBitEncoder.DEFAULT_ENABLE_RANDOM_ROTATION;
    @Builder.Default
    private final boolean enableADC = QFrameBitEncoder.DEFAULT_ENABLE_ADC;

    /**
     * Static method to generate type identifier based on ScalarQuantizationType.
     *
     * @param sqType the scalar quantization type.
     * @return A string representing the unique type identifier.
     */
    public static String generateTypeIdentifier(ScalarQuantizationType sqType) {
        return generateIdentifier(sqType.getId());
    }

    /**
     * Provides a unique type identifier for the ScalarQuantizationParams, combining the SQ type.
     * This identifier is useful for distinguishing between different configurations of scalar quantization parameters.
     *
     * @return A string representing the unique type identifier.
     */
    @Override
    public String getTypeIdentifier() {
        return generateIdentifier(sqType.getId());
    }

    /**
     * Writes the object to the output stream.
     * This method is part of the Writeable interface and is used to serialize the object.
     *
     * @param out the output stream to write the object to.
     * @throws IOException if an I/O error occurs.
     */
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(sqType.getId());
        out.writeBoolean(enableRandomRotation);
        out.writeBoolean(enableADC);
    }

    /**
     * Reads the object from the input stream.
     * This method is part of the Writeable interface and is used to deserialize the object.
     *
     * @param in the input stream to read the object from.
     * @throws IOException if an I/O error occurs.
     */
    // Constructor for deserialization
    public ScalarQuantizationParams(StreamInput in, int version) throws IOException {
        int typeId = in.readVInt();
        this.sqType = ScalarQuantizationType.fromId(typeId);
        if (Version.fromId(version).onOrAfter(Version.V_3_2_0)) {
            this.enableRandomRotation = in.readBoolean();
            this.enableADC = in.readBoolean();
        } else {
            this.enableRandomRotation = QFrameBitEncoder.DEFAULT_ENABLE_RANDOM_ROTATION;
            this.enableADC = QFrameBitEncoder.DEFAULT_ENABLE_ADC;
        }
    }

    /**
     * Generates a unique identifier for Scalar Quantization Parameters.
     *
     * <p>
     * This method constructs an identifier string by prefixing the given integer ID
     * with "ScalarQuantizationParams_". The resulting string can be used to uniquely
     * identify specific quantization parameter instances, especially when registering
     * or retrieving them in a registry or similar structure.
     * </p>
     *
     * @param id the integer ID to be used in generating the unique identifier.
     * @return a string representing the unique identifier for the quantization parameters.
     */
    private static String generateIdentifier(int id) {
        return "ScalarQuantizationParams_" + id;
    }
}
