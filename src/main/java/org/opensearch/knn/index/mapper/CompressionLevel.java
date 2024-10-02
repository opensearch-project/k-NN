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
    x4(4, "4x", null, Collections.emptySet()),
    x8(8, "8x", new RescoreContext(2.0f), Set.of(Mode.ON_DISK)),
    x16(16, "16x", new RescoreContext(3.0f), Set.of(Mode.ON_DISK)),
    x32(32, "32x", new RescoreContext(3.0f), Set.of(Mode.ON_DISK));

    // Internally, an empty string is easier to deal with them null. However, from the mapping,
    // we do not want users to pass in the empty string and instead want null. So we make the conversion here
    public static final String[] NAMES_ARRAY = new String[] {
        NOT_CONFIGURED.getName(),
        x1.getName(),
        x2.getName(),
        x4.getName(),
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
     * <p>If the {@code mode} is present in the valid {@code modesForRescore} set, the method adjusts the oversample factor based on the
     * {@code dimension} value:
     * <ul>
     *     <li>If {@code dimension} is greater than or equal to 1000, no oversampling is applied (oversample factor = 1.0).</li>
     *     <li>If {@code dimension} is greater than or equal to 768 but less than 1000, a 2x oversample factor is applied (oversample factor = 2.0).</li>
     *     <li>If {@code dimension} is less than 768, a 3x oversample factor is applied (oversample factor = 3.0).</li>
     * </ul>
     * If the {@code mode} is not present in the {@code modesForRescore} set, the method returns {@code null}.
     *
     * @param mode      The {@link Mode} for which to retrieve the {@link RescoreContext}.
     * @param dimension The dimensional value that determines the {@link RescoreContext} behavior.
     * @return          A {@link RescoreContext} with the appropriate oversample factor based on the dimension, or {@code null} if the mode
     *                  is not valid.
     */
    public RescoreContext getDefaultRescoreContext(Mode mode, int dimension) {
        if (modesForRescore.contains(mode)) {
            // Adjust RescoreContext based on dimension
            if (dimension >= RescoreContext.DIMENSION_THRESHOLD_1000) {
                // No oversampling for dimensions >= 1000
                return RescoreContext.builder().oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_1000).build();
            } else if (dimension >= RescoreContext.DIMENSION_THRESHOLD_768) {
                // 2x oversampling for dimensions >= 768 but < 1000
                return RescoreContext.builder().oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_768).build();
            } else {
                // 3x oversampling for dimensions < 768
                return RescoreContext.builder().oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_BELOW_768).build();
            }
        }
        return null;
    }

}
