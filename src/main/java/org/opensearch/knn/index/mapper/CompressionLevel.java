/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.core.common.Strings;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Locale;

/**
 * Enum representing the compression level for float vectors. Compression in this sense refers to compressing a
 * full precision value into a smaller number of bits. For instance. "16x" compression would mean that 2 bits would
 * need to be used to represent a 32-bit floating point number.
 */
@AllArgsConstructor
public enum CompressionLevel {
    NOT_CONFIGURED(-1, "", null),
    x1(1, "1x", null),
    x2(2, "2x", null),
    x4(4, "4x", new RescoreContext(1.0f)),
    x8(8, "8x", new RescoreContext(1.5f)),
    x16(16, "16x", new RescoreContext(2.0f)),
    x32(32, "32x", new RescoreContext(2.0f));

    // Internally, an empty string is easier to deal with them null. However, from the mapping,
    // we do not want users to pass in the empty string and instead want null. So we make the conversion herex
    public static final String[] NAMES_ARRAY = new String[] {
        NOT_CONFIGURED.getName(),
        x1.getName(),
        x2.getName(),
        x8.getName(),
        x16.getName(),
        x32.getName() };

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
    public static CompressionLevel fromName(String name) {
        if (Strings.isEmpty(name)) {
            return NOT_CONFIGURED;
        }
        for (CompressionLevel config : CompressionLevel.values()) {
            if (config.getName() != null && config.getName().equals(name)) {
                return config;
            }
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid compression level: \"[%s]\"", name));
    }

    private final int compressionLevel;
    @Getter
    private final String name;
    @Getter
    private final RescoreContext defaultRescoreContext;

    /**
     * Gets the number of bits used to represent a float in order to achieve this compression. For instance, for
     * 32x compression, each float would need to be encoded in a single bit.
     *
     * @return number of bits to represent a float at this compression level
     */
    public int numBitsForFloat32() {
        if (this == NOT_CONFIGURED) {
            return DEFAULT.numBitsForFloat32();
        }

        return (Float.BYTES * Byte.SIZE) / compressionLevel;
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
