/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnSearchStrategy;

public class WarmUpCollector implements KnnCollector {
    public final static WarmUpCollector INSTANCE = new WarmUpCollector();

    public static boolean isWarmUpRequest(final KnnCollector object) {
        return object == INSTANCE;
    }

    @Override
    public boolean earlyTerminated() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void incVisitedCount(int count) {
        throw new UnsupportedOperationException();
    }

    @Override
    public long visitedCount() {
        throw new UnsupportedOperationException();
    }

    @Override
    public long visitLimit() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int k() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean collect(int docId, float similarity) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float minCompetitiveSimilarity() {
        throw new UnsupportedOperationException();
    }

    @Override
    public TopDocs topDocs() {
        throw new UnsupportedOperationException();
    }

    @Override
    public KnnSearchStrategy getSearchStrategy() {
        throw new UnsupportedOperationException();
    }
}
