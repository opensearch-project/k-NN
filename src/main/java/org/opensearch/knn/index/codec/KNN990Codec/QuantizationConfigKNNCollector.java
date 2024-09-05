/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

/**
 * Collector used for passing the quantization state during query flow.
 */
@Setter
@Getter
public class QuantizationConfigKNNCollector implements KnnCollector {

    private QuantizationState quantizationState;

    private final String NATIVE_ENGINE_SEARCH_ERROR_MESSAGE = "Search functionality using codec is not supported with Native Engine Reader";

    @Override
    public boolean earlyTerminated() {
        throw new UnsupportedOperationException(NATIVE_ENGINE_SEARCH_ERROR_MESSAGE);
    }

    @Override
    public void incVisitedCount(int i) {
        throw new UnsupportedOperationException(NATIVE_ENGINE_SEARCH_ERROR_MESSAGE);
    }

    @Override
    public long visitedCount() {
        throw new UnsupportedOperationException(NATIVE_ENGINE_SEARCH_ERROR_MESSAGE);
    }

    @Override
    public long visitLimit() {
        throw new UnsupportedOperationException(NATIVE_ENGINE_SEARCH_ERROR_MESSAGE);
    }

    @Override
    public int k() {
        throw new UnsupportedOperationException(NATIVE_ENGINE_SEARCH_ERROR_MESSAGE);
    }

    @Override
    public boolean collect(int i, float v) {
        throw new UnsupportedOperationException(NATIVE_ENGINE_SEARCH_ERROR_MESSAGE);
    }

    @Override
    public float minCompetitiveSimilarity() {
        throw new UnsupportedOperationException(NATIVE_ENGINE_SEARCH_ERROR_MESSAGE);
    }

    @Override
    public TopDocs topDocs() {
        throw new UnsupportedOperationException(NATIVE_ENGINE_SEARCH_ERROR_MESSAGE);
    }
}
