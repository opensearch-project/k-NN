/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationParams;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Locale;

/**
 * The SQParams class represents the parameters specific to scalar quantization (SQ).
 * This class implements the QuantizationParams interface and includes the type of scalar quantization.
 */
@Getter
@AllArgsConstructor
@NoArgsConstructor // No-argument constructor for deserialization
@EqualsAndHashCode
public class ScalarQuantizationParams implements QuantizationParams {
    private ScalarQuantizationType sqType;
    private static final long serialVersionUID = 1L; // Version ID for serialization

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
     * Serializes the SQParams object to an external output.
     * This method writes the scalar quantization type to the output stream.
     *
     * @param out the ObjectOutput to write the object to.
     * @throws IOException if an I/O error occurs during serialization.
     */
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        // The version is already written by the parent state class, no need to write it here again
        // Retrieve the current version from VersionContext
        // This context will be used by other classes involved in the serialization process.
        // Example:
        // int version = VersionContext.getVersion(); // Get the current version from VersionContext
        // Any Version Specific logic can be wriiten based on Version
        out.writeObject(sqType);
    }

    /**
     * Deserializes the SQParams object from an external input with versioning.
     * This method reads the scalar quantization type and new field from the input stream based on the version.
     *
     * @param in the ObjectInput to read the object from.
     * @throws IOException if an I/O error occurs during deserialization.
     * @throws ClassNotFoundException if the class of the serialized object cannot be found.
     */
    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        // The version is already read by the parent state class and set in VersionContext
        // Retrieve the current version from VersionContext to handle version-specific deserialization logic
        // int versionId = VersionContext.getVersion();
        // Version version = Version.fromId(versionId);

        sqType = (ScalarQuantizationType) in.readObject();

        // Add version-specific deserialization logic
        // For example, if new fields are added in a future version, handle them here
        // This section contains conditional logic to handle different versions appropriately.
        // Example:
        // if (version.onOrAfter(Version.V_1_0_0) && version.before(Version.V_2_0_0)) {
        // // Handle logic for versions between 1.0.0 and 2.0.0
        // // Example: Read additional fields introduced in version 1.0.0
        // // newField = in.readInt();
        // } else if (version.onOrAfter(Version.V_2_0_0)) {
        // // Handle logic for versions 2.0.0 and above
        // // Example: Read additional fields introduced in version 2.0.0
        // // anotherNewField = in.readFloat();
        // }
    }

    /**
     * Provides a unique type identifier for the SQParams, combining the SQ type.
     * This identifier is useful for distinguishing between different configurations of scalar quantization parameters.
     *
     * @return A string representing the unique type identifier.
     */
    @Override
    public String getTypeIdentifier() {
        return generateIdentifier(sqType.getId());
    }

    private static String generateIdentifier(int id) {
        return String.format(Locale.ROOT, "ScalarQuantizationParams_%d", id);
    }
}
