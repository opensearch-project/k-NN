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

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Class provides params for Lucene104HnswScalarQuantizedVectorsFormat
 */
@Getter
public class KNN1040ScalarQuantizedVectorsFormatParams extends KNNVectorsFormatParams {
    private static final Set<String> SUPPORTED_ENCODERS = Set.of(ENCODER_SQ);
    private String encoderName;
    private ScalarEncoding bitEncoding;

    public KNN1040ScalarQuantizedVectorsFormatParams(Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        super(params, defaultMaxConnections, defaultBeamWidth);
        initFields(params);
    }

    private void initFields(Map<String, Object> params) {
        if (params.get(METHOD_ENCODER_PARAMETER) == null) {
            return;
        }

        if ((params.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext) == false) {
            return;
        }

        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) params.get(METHOD_ENCODER_PARAMETER);
        this.encoderName = encoderMethodComponentContext.getName();
        Map<String, Object> encoderParams = encoderMethodComponentContext.getParameters();
        if (this.encoderName != null) this.initBits(encoderParams);
    }

    @Override
    public boolean validate(final Map<String, Object> params) {
        return this.encoderName != null && SUPPORTED_ENCODERS.contains(this.encoderName);
    }

    private ScalarEncoding getOrDefaultBitsForEncoder(final Map<String, Object> params, String bitsKey, int defaultBits) {
        if (params != null && params.containsKey(bitsKey)) {
            return ScalarEncoding.fromNumBits((int) params.get(bitsKey));
        }
        return ScalarEncoding.fromNumBits(defaultBits);
    }

    private void initBits(final Map<String, Object> params) {
        this.bitEncoding = getOrDefaultBitsForEncoder(params, LUCENE_SQ_BITS, OPTIMIZED_SCALAR_QUANTIZER_DEFAULT_BITS);
    }
}
