/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import lombok.Getter;
import org.opensearch.knn.common.KNNConstants;

import java.util.Map;

/**
 * Class provides params for LuceneHNSWVectorsFormat
 */
@Getter
public class KNNVectorsFormatParams {
    private int maxConnections;
    private int beamWidth;

    public KNNVectorsFormatParams(final Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        initMaxConnections(params, defaultMaxConnections);
        initBeamWidth(params, defaultBeamWidth);
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
