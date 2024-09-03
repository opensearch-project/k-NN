/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.core.common.Strings;

import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Enum representing the intended workload optimization a user wants their k-NN system to have. Based on this value,
 * default parameter resolution will be determined.
 */
@Getter
@AllArgsConstructor
public enum Mode {
    NOT_CONFIGURED(""),
    IN_MEMORY("in_memory"),
    ON_DISK("on_disk");

    // Internally, an empty string is easier to deal with them null. However, from the mapping,
    // we do not want users to pass in the empty string and instead want null. So we make the conversion herex
    static final String[] NAMES_ARRAY = Arrays.stream(Mode.values())
        .map(mode -> mode == NOT_CONFIGURED ? null : mode.getName())
        .collect(Collectors.toList())
        .toArray(new String[0]);

    private static final Mode DEFAULT = IN_MEMORY;

    /**
     * Convert a string to a Mode enum value
     *
     * @param name String value to convert
     * @return Mode enum value
     */
    public static Mode fromName(String name) {
        if (Strings.isEmpty(name)) {
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
