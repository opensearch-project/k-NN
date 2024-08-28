/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;

@Setter
@Getter
public class QuantizationConfigKNNCollector implements KnnCollector {

    private SegmentReadState segmentReadState;

    @Override
    public boolean earlyTerminated() {
        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    @Override
    public void incVisitedCount(int i) {
        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    @Override
    public long visitedCount() {
        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    @Override
    public long visitLimit() {
        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    @Override
    public int k() {
        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    @Override
    public boolean collect(int i, float v) {
        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    @Override
    public float minCompetitiveSimilarity() {
        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }

    @Override
    public TopDocs topDocs() {
        throw new UnsupportedOperationException("Search functionality using codec is not supported with Native Engine Reader");
    }
}
