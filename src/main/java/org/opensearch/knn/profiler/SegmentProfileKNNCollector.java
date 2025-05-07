/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profiler;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;

/**
 * Segment profiler collector for KNN plugin which is used to
 * collect profiling information for a segment.
 */
@Setter
@Getter
public class SegmentProfileKNNCollector implements KnnCollector {

    private SegmentProfilerState segmentProfilerState;

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
