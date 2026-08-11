/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.core.common.Strings;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Collections;
import java.util.Locale;
import java.util.Set;

/**
 * Enum representing the compression level for float vectors. Compression in this sense refers to compressing a
 * full precision value into a smaller number of bits. For instance. "16x" compression would mean that 2 bits would
 * need to be used to represent a 32-bit floating point number.
 */
@AllArgsConstructor
public enum CompressionLevel {
    NOT_CONFIGURED(-1, "", null, Collections.emptySet()),
    x1(1, "1x", null, Collections.emptySet()),
    x2(2, "2x", null, Collections.emptySet()),
    x4(4, "4x", RescoreContext.builder().oversampleFactor(1.0f).userProvided(false).build(), Set.of(Mode.ON_DISK)),
    x8(8, "8x", RescoreContext.builder().oversampleFactor(2.0f).userProvided(false).build(), Set.of(Mode.ON_DISK)),
    x16(16, "16x", RescoreContext.builder().oversampleFactor(3.0f).userProvided(false).build(), Set.of(Mode.ON_DISK)),
    x32(32, "32x", RescoreContext.builder().oversampleFactor(3.0f).userProvided(false).build(), Set.of(Mode.ON_DISK)),
    x64(64, "64x", RescoreContext.builder().oversampleFactor(5.0f).userProvided(false).build(), Set.of(Mode.ON_DISK));

    public static final CompressionLevel MAX_COMPRESSION_LEVEL = CompressionLevel.x64;

    /**
     * Default is set to 1x and is a noop
     */
    private static final CompressionLevel DEFAULT = x1;
    private static final float FLAT_OVERSAMPLE_FACTOR = 2.0f;

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
    /**
     * Pre-configured default {@link RescoreContext} for this compression level, without any
     * conditional logic. Null for levels that do not require rescoring (NOT_CONFIGURED, x1, x2).
     */
    @Getter
    private final RescoreContext defaultRescoreContext;
    private final Set<Mode> modesForRescore;

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

    /**
     * Whether the given mode is valid for rescoring at this compression level.
     */
    public boolean isModeValidForRescore(Mode mode) {
        return modesForRescore.contains(mode);
    }

}
