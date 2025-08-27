/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import org.opensearch.knn.index.engine.MethodComponentContext;
import java.util.Map;
import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Class provides params for Lucene102HnswBinaryQuantizedVectorsFormat
 */
public class KNNBBQVectorsFormatParams extends KNNVectorsFormatParams {

    public KNNBBQVectorsFormatParams(Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        super(params, defaultMaxConnections, defaultBeamWidth);
        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        Map<String, Object> bbqEncoderParams = encoderMethodComponentContext.getParameters();
    }

    @Override
    public boolean validate(Map<String, Object> params) {
        if (params.get(METHOD_ENCODER_PARAMETER) == null) {
            return false;
        }

        if (!(params.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext)) {
            return false;
        }

        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        return ENCODER_BBQ.equals(encoderMethodComponentContext.getName());
    }
}
