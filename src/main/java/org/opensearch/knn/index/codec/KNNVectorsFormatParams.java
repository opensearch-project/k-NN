/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.knn.common.KNNConstants;

import java.util.Map;

/**
 * Class provides params for LuceneHNSWVectorsFormat
 */
@NoArgsConstructor
@Getter
public class KNNVectorsFormatParams {
    private int maxConnections;
    private int beamWidth;

    protected boolean validate(final Map<String, Object> params) {
        return false;
    }

    protected void initialize(final Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        this.maxConnections = getMaxConnections(params, defaultMaxConnections);
        this.beamWidth = getBeamWidth(params, defaultBeamWidth);
    }

    private int getMaxConnections(final Map<String, Object> params, int defaultMaxConnections) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_M)) {
            return (int) params.get(KNNConstants.METHOD_PARAMETER_M);
        }
        return defaultMaxConnections;
    }

    private int getBeamWidth(final Map<String, Object> params, int defaultBeamWidth) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION)) {
            return (int) params.get(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION);
        }
        return defaultBeamWidth;
    }
}
