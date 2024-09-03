/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;

import java.util.Locale;

/**
 * Enum representing the intended workload optimization a user wants their k-NN system to have. Based on this value,
 * default parameter resolution will be determined.
 */
@AllArgsConstructor
public enum Mode {
    NOT_CONFIGURED(null),
    IN_MEMORY("in_memory"),
    ON_DISK("on_disk");

    private static final Mode DEFAULT = IN_MEMORY;

    /**
     * Convert a string to a Mode enum value
     *
     * @param name String value to convert
     * @return Mode enum value
     */
    public static Mode fromString(String name) {
        if (name == null) {
            return NOT_CONFIGURED;
        }

        if (IN_MEMORY.name.equalsIgnoreCase(name)) {
            return IN_MEMORY;
        }

        if (ON_DISK.name.equalsIgnoreCase(name)) {
            return ON_DISK;
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid mode: \"[%s]\"", name));
    }

    private final String name;

    @Override
    public String toString() {
        return name;
    }

    /**
     * Utility method that checks if mode is configured.
     *
     * @param mode Mode to check
     * @return true if mode is configured, false otherwise
     */
    public static boolean isConfigured(Mode mode) {
        return mode != null && mode != NOT_CONFIGURED;
    }
}
