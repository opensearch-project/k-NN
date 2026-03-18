/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import lombok.Getter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_BBQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_BBQ_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Class provides params for Lucene104HnswScalarQuantizedVectorsFormat
 */
@Getter
public class KNN1040ScalarQuantizedVectorsFormatParams extends KNNVectorsFormatParams {
    private static final Set<String> SUPPORTED_ENCODERS = Set.of(ENCODER_BBQ);
    private String encoderName;
    private ScalarEncoding bitEncoding;

    public KNN1040ScalarQuantizedVectorsFormatParams(Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        super(params, defaultMaxConnections, defaultBeamWidth);
        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        Map<String, Object> encoderParams = encoderMethodComponentContext.getParameters();
        this.encoderName = this.resolveEncoderName(params);
        if (this.encoderName != null) this.initBits(encoderParams);
    }

    private String resolveEncoderName(Map<String, Object> params) {
        if (params.get(METHOD_ENCODER_PARAMETER) == null) {
            return null;
        }

        if ((params.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext) == false) {
            return null;
        }

        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        return encoderMethodComponentContext.getName();
    }

    @Override
    public boolean validate(final Map<String, Object> params) {
        return encoderName != null && SUPPORTED_ENCODERS.contains(encoderName);
    }

    private ScalarEncoding getOrDefaultBitsForEncoder(final Map<String, Object> params, String bitsKey, int defaultBits) {
        if (params != null && params.containsKey(bitsKey)) {
            return ScalarEncoding.fromNumBits((int) params.get(bitsKey));
        }
        return ScalarEncoding.fromNumBits(defaultBits);
    }

    private void initBits(final Map<String, Object> params) {
        this.bitEncoding = getOrDefaultBitsForEncoder(params, LUCENE_BBQ_BITS, LUCENE_BBQ_DEFAULT_BITS);
    }
}
