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
    @Builder.Default
    private Mode mode = Mode.NOT_CONFIGURED;
    @Builder.Default
    private CompressionLevel compressionLevel = CompressionLevel.NOT_CONFIGURED;

    public static final KNNMethodConfigContext EMPTY = KNNMethodConfigContext.builder().build();
}
