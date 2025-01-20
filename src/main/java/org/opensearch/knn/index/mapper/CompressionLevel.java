/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.Version;
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
    // TODO: Revisit and fix this
    x4(4, "4x", new RescoreContext(2.0f, false, true), Set.of(Mode.ON_DISK)),
    x8(8, "8x", new RescoreContext(2.0f, false, true), Set.of(Mode.ON_DISK)),
    x16(16, "16x", new RescoreContext(3.0f, false, true), Set.of(Mode.ON_DISK)),
    x32(32, "32x", new RescoreContext(3.0f, false, true), Set.of(Mode.ON_DISK)),
    x64(64, "64x", new RescoreContext(5.0f, false, true), Set.of(Mode.ON_DISK));

    public static final CompressionLevel MAX_COMPRESSION_LEVEL = CompressionLevel.x64;

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
     * Returns the appropriate {@link RescoreContext} based on the given {@code mode} and {@code dimension}.
     *
     * <p>If the {@code mode} is present in the valid {@code modesForRescore} set, the method checks the value of
     * {@code dimension}:
     * <ul>
     *     <li>If {@code dimension} is less than or equal to 1000, it returns a {@link RescoreContext} with an
     *         oversample factor of 5.0f.</li>
     *     <li>If {@code dimension} is greater than 1000, it returns the default {@link RescoreContext} associated with
     *         the {@link CompressionLevel}. If no default is set, it falls back to {@link RescoreContext#getDefault()}.</li>
     * </ul>
     * If the {@code mode} is not valid, the method returns {@code null}.
     *
     * @param mode      The {@link Mode} for which to retrieve the {@link RescoreContext}.
     * @param dimension The dimensional value that determines the {@link RescoreContext} behavior.
     * @param indexVersionCreated OpenSearch cluster version in which the index was created
     * @return          A {@link RescoreContext} with an oversample factor of 5.0f if {@code dimension} is less than
     *                  or equal to 1000, the default {@link RescoreContext} if greater, or {@code null} if the mode
     *                  is invalid.
     */
    public RescoreContext getDefaultRescoreContext(Mode mode, int dimension, Version indexVersionCreated) {

        // x4 compression was supported by Lucene engine before version 2.19.0 and there is no default rescore context
        if (compressionLevel == CompressionLevel.x4.compressionLevel && indexVersionCreated.before(Version.V_2_19_0)) {
            return null;
        }
        if (modesForRescore.contains(mode)) {
            // Adjust RescoreContext based on dimension
            if (dimension <= RescoreContext.DIMENSION_THRESHOLD) {
                // For dimensions <= 1000, return a RescoreContext with 5.0f oversample factor
                return RescoreContext.builder()
                    .oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD)
                    .userProvided(false)
                    .build();
            } else {
                return defaultRescoreContext;
            }
        }
        return null;
    }

}
