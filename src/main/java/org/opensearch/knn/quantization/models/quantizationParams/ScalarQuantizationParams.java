/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationParams;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;

/**
 * The ScalarQuantizationParams class represents the parameters specific to scalar quantization (SQ).
 * This class implements the QuantizationParams interface and includes the type of scalar quantization.
 */
@Getter
@AllArgsConstructor
@NoArgsConstructor // No-argument constructor for deserialization
@EqualsAndHashCode
public class ScalarQuantizationParams implements QuantizationParams {
    private ScalarQuantizationType sqType;

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

    private static String generateIdentifier(int id) {
        return "ScalarQuantizationParams_" + id;
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
    }

    /**
     * Reads the object from the input stream.
     * This method is part of the Writeable interface and is used to deserialize the object.
     *
     * @param in the input stream to read the object from.
     * @throws IOException if an I/O error occurs.
     */
    public ScalarQuantizationParams(StreamInput in, int version) throws IOException {
        int typeId = in.readVInt();
        this.sqType = ScalarQuantizationType.fromId(typeId);
    }
}
