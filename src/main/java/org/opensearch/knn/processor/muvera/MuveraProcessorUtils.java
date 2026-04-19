/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import java.util.Map;

/**
 * Shared utility methods for MUVERA ingest and search processors.
 */
final class MuveraProcessorUtils {

    private MuveraProcessorUtils() {}

    /**
     * Reads a long property from the processor config map, with a default value.
     * Removes the property from the map (same behavior as ConfigurationUtils.readIntProperty).
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
