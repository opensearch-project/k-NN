/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

import java.util.Map;

/**
 * Shared utility methods for the MUVERA ingest and search request processors.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
final class MuveraProcessorUtils {

    /**
     * Reads a {@code long} property from a processor config map, falling back to a default
     * when the property is absent. The property is removed from the map (matching the behavior
     * of {@code ConfigurationUtils.readIntProperty}).
     *
     * @param processorType processor type, used in the exception message for diagnostics
     * @param processorTag  processor tag (the name the user assigned in the pipeline)
     * @param config        processor config map (mutated: the read property is removed)
     * @param propertyName  config key to read
     * @param defaultValue  value returned when the property is absent
     * @return the parsed long, or {@code defaultValue} if the property was absent
     * @throws IllegalArgumentException if the property is present but not a valid long
     */
    static long readLongProperty(
        String processorType,
        String processorTag,
        Map<String, Object> config,
        String propertyName,
        long defaultValue
    ) {
        Object value = config.remove(propertyName);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Long.parseLong(value.toString());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                "["
                    + processorType
                    + "] processor ["
                    + processorTag
                    + "] property ["
                    + propertyName
                    + "] is not a valid long: ["
                    + value
                    + "]"
            );
        }
    }
}
