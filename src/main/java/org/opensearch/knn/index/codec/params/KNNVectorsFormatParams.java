/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import lombok.Getter;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.util.Map;

/**
 * Class provides params for LuceneHNSWVectorsFormat
 */
@Getter
public class KNNVectorsFormatParams {
    private int maxConnections;
    private int beamWidth;
    private final SpaceType spaceType;
    private final VectorDataType vectorDataType;

    public KNNVectorsFormatParams(final Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        this(params, defaultMaxConnections, defaultBeamWidth, SpaceType.UNDEFINED, VectorDataType.DEFAULT);
    }

    public KNNVectorsFormatParams(
        final Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth,
        SpaceType spaceType,
        VectorDataType vectorDataType
    ) {
        initMaxConnections(params, defaultMaxConnections);
        initBeamWidth(params, defaultBeamWidth);
        this.spaceType = spaceType;
        this.vectorDataType = vectorDataType;
    }

    public boolean validate(final Map<String, Object> params) {
        return true;
    }

    private void initMaxConnections(final Map<String, Object> params, int defaultMaxConnections) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_M)) {
            this.maxConnections = (int) params.get(KNNConstants.METHOD_PARAMETER_M);
            return;
        }
        this.maxConnections = defaultMaxConnections;
    }

    private void initBeamWidth(final Map<String, Object> params, int defaultBeamWidth) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION)) {
            this.beamWidth = (int) params.get(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION);
            return;
        }
        this.beamWidth = defaultBeamWidth;
    }
}
