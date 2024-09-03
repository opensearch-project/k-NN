/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodContext;

/**
 * Utility class to store the original mapping parameters for a KNNVectorFieldMapper. These parameters need to be
 * kept around for when a {@link KNNVectorFieldMapper} is built from merge
 */
@Getter
@RequiredArgsConstructor
public final class OriginalMappingParameters {
    private final VectorDataType vectorDataType;
    private final int dimension;
    private final KNNMethodContext knnMethodContext;
    /**
     * Resolved method context is used in order to handle legacy case when the user does not pass in a knnMethodContext
     * but one is created from the context in the settings. By default, it will match the passed in value
     */
    @Setter
    private KNNMethodContext resolvedKnnMethodContext;
    private final String mode;
    private final String compressionLevel;
    private final String modelId;

    /**
     * Initialize the parameters from the builder
     *
     * @param builder The builder to initialize from
     */
    public OriginalMappingParameters(KNNVectorFieldMapper.Builder builder) {
        this.vectorDataType = builder.vectorDataType.get();
        this.knnMethodContext = builder.knnMethodContext.get();
        this.resolvedKnnMethodContext = builder.knnMethodContext.get();
        this.dimension = builder.dimension.get();
        this.mode = builder.mode.get();
        this.compressionLevel = builder.compressionLevel.get();
        this.modelId = builder.modelId.get();
    }

    /**
     * Determine if the mapping used the legacy mechanism to setup the index. The legacy mechanism is used if
     * the index is created only by specifying the dimension. If this is the case, the constructed parameters
     * need to be collected from the index settings
     *
     * @return true if the mapping used the legacy mechanism, false otherwise
     */
    public boolean isLegacyMapping() {
        if (knnMethodContext != null) {
            return false;
        }

        if (vectorDataType != VectorDataType.DEFAULT) {
            return false;
        }

        if (modelId != null || dimension == -1) {
            return false;
        }

        return mode == null && compressionLevel == null;
    }
}
