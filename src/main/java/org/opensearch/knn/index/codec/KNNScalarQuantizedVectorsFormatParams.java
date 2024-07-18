/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.knn.index.MethodComponentContext;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_COMPRESS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_CONFIDENCE_INTERVAL;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Class provides params for LuceneHnswScalarQuantizedVectorsFormat
 */
@Getter
@NoArgsConstructor
public class KNNScalarQuantizedVectorsFormatParams extends KNNVectorsFormatParams {
    private float confidenceInterval;
    private int bits;
    private boolean compressFlag;

    @Override
    protected boolean validate(Map<String, Object> params) {
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

    @Override
    protected void initialize(Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        super.initialize(params, defaultMaxConnections, defaultBeamWidth);
        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        Map<String, Object> sqEncoderParams = encoderMethodComponentContext.getParameters();
        this.confidenceInterval = getConfidenceInterval(sqEncoderParams);
        this.bits = getBits(sqEncoderParams);
        this.compressFlag = getCompressFlag(sqEncoderParams);
    }

    private Float getConfidenceInterval(final Map<String, Object> params) {

        if (params != null && params.containsKey(LUCENE_SQ_CONFIDENCE_INTERVAL)) {
            if (params.get(LUCENE_SQ_CONFIDENCE_INTERVAL).equals(0)) return Float.valueOf(0);

            return ((Double) params.get(LUCENE_SQ_CONFIDENCE_INTERVAL)).floatValue();

        }
        return null;
    }

    private int getBits(final Map<String, Object> params) {
        if (params != null && params.containsKey(LUCENE_SQ_BITS)) {
            return (int) params.get(LUCENE_SQ_BITS);
        }
        return LUCENE_SQ_DEFAULT_BITS;
    }

    private boolean getCompressFlag(final Map<String, Object> params) {
        if (params != null && params.containsKey(LUCENE_SQ_COMPRESS)) {
            return (boolean) params.get(LUCENE_SQ_COMPRESS);
        }
        return false;
    }
}
