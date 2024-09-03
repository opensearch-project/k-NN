/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;

import java.util.Locale;

/**
 * Enum representing the compression level for float vectors. Compression in this sense refers to compressing a
 * full precision value into a smaller number of bits. For instance. "16x" compression would mean that 2 bits would
 * need to be used to represent a 32-bit floating point number.
 */
@AllArgsConstructor
public enum CompressionLevel {
    NOT_CONFIGURED(-1),
    x1(1),
    x2(2),
    x4(4),
    x8(8),
    x16(16),
    x32(32);

    /**
     * Default is set to 1x and is a noop
     */
    private static final CompressionLevel DEFAULT = x1;

    /**
     * Get the compression level from a string representation. The format for the string should be "Nx", where N is
     * the factor by which compression should take place
     *
     * @param name String representation of the compression level
     * @return CompressionLevel enum value
     */
    public static CompressionLevel fromString(String name) {
        if (name == null) {
            return NOT_CONFIGURED;
        }
        for (CompressionLevel config : CompressionLevel.values()) {
            if (config.toString() != null && config.toString().equals(name)) {
                return config;
            }
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid compression level: \"[%s]\"", name));
    }

    private final int compressionLevel;

    @Override
    public String toString() {
        if (this == NOT_CONFIGURED) {
            return null;
        }
        return compressionLevel + "x";
    }

    /**
     * Gets the number of bits used to represent a float in order to achieve this compression. For instance, for
     * 32x compression, each float would need to be encoded in a single bit.
     *
     * @return number of bits to represent a float at this compression level
     */
    public int numBitsForFloat() {
        if (this == NOT_CONFIGURED) {
            return DEFAULT.numBitsForFloat();
        }

        return (Float.BYTES * Byte.SIZE) * compressionLevel;
    }

    /**
     * Utility method that checks if compression is configured.
     *
     * @param compressionLevel Compression to check
     * @return true if compression is configured, false otherwise
     */
    public static boolean isConfigured(CompressionLevel compressionLevel) {
        return compressionLevel != null && compressionLevel != NOT_CONFIGURED;
    }
}
