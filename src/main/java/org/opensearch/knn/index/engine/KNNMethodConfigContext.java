/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

/**
 * This object provides additional context that the user does not provide when {@link KNNMethodContext} is
 * created via parsing. The values in this object need to be dynamically set and calling code needs to handle
 * the possibility that the values have not been set.
 */
@Setter
@Getter
@Builder
@EqualsAndHashCode
public final class KNNMethodConfigContext {
    private VectorDataType vectorDataType;
    private Integer dimension;
    private Version versionCreated;
    @Deprecated
    @Builder.Default
    private Mode mode = Mode.NOT_CONFIGURED;
    @Builder.Default
    private CompressionLevel compressionLevel = CompressionLevel.NOT_CONFIGURED;
    /**
     * Snapshot of the compression level as it was configured by the user, captured at construction time
     * before method resolution overwrites {@link #compressionLevel} with an encoder-derived value (e.g.
     * binary encoder bits=1 resolves to x32). Mode derivation must be based on this value: a user asking
     * for compression_level=32x implies on_disk behavior, but an encoder that internally maps to x32
     * does not.
     */
    @EqualsAndHashCode.Exclude
    private final CompressionLevel userConfiguredCompressionLevel;

    public static final KNNMethodConfigContext EMPTY = KNNMethodConfigContext.builder().build();

    KNNMethodConfigContext(
        VectorDataType vectorDataType,
        Integer dimension,
        Version versionCreated,
        Mode mode,
        CompressionLevel compressionLevel,
        CompressionLevel userConfiguredCompressionLevel
    ) {
        this.vectorDataType = vectorDataType;
        this.dimension = dimension;
        this.versionCreated = versionCreated;
        this.mode = mode;
        this.compressionLevel = compressionLevel;
        // At build time, compressionLevel still holds the user's value since resolution has not run yet.
        // Callers reconstructing a context post-resolution (e.g. mapper merge) pass the user value explicitly.
        this.userConfiguredCompressionLevel = userConfiguredCompressionLevel == null ? compressionLevel : userConfiguredCompressionLevel;
    }

    /**
     * Derives the appropriate {@link Mode} from the given {@link CompressionLevel}.
     * Starting with {@code KNNConstants.KNN_DEFAULT_COMPRESSION_FLIP_VERSION} (V_3_8_0), mode is derived
     * from compression rather than being an independent axis.
     *
     * @param compressionLevel the compression level to derive mode from
     * @return the derived mode
     */
    public static Mode deriveMode(CompressionLevel compressionLevel) {
        if (!CompressionLevel.isConfigured(compressionLevel)) {
            return Mode.NOT_CONFIGURED;
        }
        if (compressionLevel == CompressionLevel.x1 || compressionLevel == CompressionLevel.x2) {
            return Mode.IN_MEMORY;
        }
        return Mode.ON_DISK;
    }
}
