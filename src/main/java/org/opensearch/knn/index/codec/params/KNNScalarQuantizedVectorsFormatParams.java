/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import lombok.Getter;
import org.opensearch.knn.index.MethodComponentContext;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Class provides params for LuceneHnswScalarQuantizedVectorsFormat
 */
@Getter
public class KNNScalarQuantizedVectorsFormatParams extends KNNVectorsFormatParams {
    private Float confidenceInterval;
    private int bits;
    private boolean compressFlag;

    public KNNScalarQuantizedVectorsFormatParams(Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        super(params, defaultMaxConnections, defaultBeamWidth);
        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        Map<String, Object> sqEncoderParams = encoderMethodComponentContext.getParameters();
        this.initConfidenceInterval(sqEncoderParams);
        this.initBits(sqEncoderParams);
        this.initCompressFlag();
    }

    @Override
    public boolean validate(Map<String, Object> params) {
        if (params.get(METHOD_ENCODER_PARAMETER) == null) {
            return false;
        }

        // Validate if the object is of type MethodComponentContext before casting it later
        if (!(params.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext)) {
            return false;
        }
        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        if (!ENCODER_SQ.equals(encoderMethodComponentContext.getName())) {
            return false;
        }

        return true;
    }

    private void initConfidenceInterval(final Map<String, Object> params) {

        if (params != null && params.containsKey(LUCENE_SQ_CONFIDENCE_INTERVAL)) {
            if (params.get(LUCENE_SQ_CONFIDENCE_INTERVAL).equals(0)) {
                this.confidenceInterval = (float) 0;
                return;
            }
            this.confidenceInterval = ((Double) params.get(LUCENE_SQ_CONFIDENCE_INTERVAL)).floatValue();
            return;
        }

        // If confidence_interval is not provided by user, then it will be set with a default value as null so that
        // it will be computed later in Lucene based on the dimension of the vector as 1 - 1/(1 + d)
        this.confidenceInterval = null;
    }

    private void initBits(final Map<String, Object> params) {
        if (params != null && params.containsKey(LUCENE_SQ_BITS)) {
            this.bits = (int) params.get(LUCENE_SQ_BITS);
            return;
        }
        this.bits = LUCENE_SQ_DEFAULT_BITS;
    }

    private void initCompressFlag() {
        this.compressFlag = true;
    }
}
