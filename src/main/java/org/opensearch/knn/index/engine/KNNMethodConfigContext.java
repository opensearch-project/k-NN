/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
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
@AllArgsConstructor
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
     * Snapshot of the compression level as it was configured by the user, taken before method resolution
     * overwrites {@link #compressionLevel} with an encoder-derived value (e.g. binary encoder bits=1
     * resolves to x32). A null value means resolution has not overwritten the compression level, so
     * {@link #compressionLevel} still holds the user-configured value.
     */
    @EqualsAndHashCode.Exclude
    private CompressionLevel userConfiguredCompressionLevel;

    public static final KNNMethodConfigContext EMPTY = KNNMethodConfigContext.builder().build();

    /**
     * Sets the resolved compression level. On the first call, snapshots the current (user-configured)
     * compression level so that consumers can distinguish compression the user explicitly requested from
     * compression derived from the encoder during method resolution.
     *
     * @param compressionLevel the resolved compression level
     */
    public void setCompressionLevel(CompressionLevel compressionLevel) {
        if (userConfiguredCompressionLevel == null) {
            userConfiguredCompressionLevel = this.compressionLevel;
        }
        this.compressionLevel = compressionLevel;
    }

    /**
     * Returns the compression level explicitly configured by the user, which may differ from
     * {@link #getCompressionLevel()} when resolution derived the compression from the encoder. Mode
     * derivation must be based on this value: a user asking for compression_level=32x implies on_disk
     * behavior, but an encoder that internally maps to x32 does not.
     *
     * @return the user-configured compression level
     */
    public CompressionLevel getUserConfiguredCompressionLevel() {
        return userConfiguredCompressionLevel == null ? compressionLevel : userConfiguredCompressionLevel;
    }

    /**
     * Derives the appropriate {@link Mode} from the given {@link CompressionLevel}.
     * For V_3_7_0+ indices, mode is derived from compression rather than being an independent axis.
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
