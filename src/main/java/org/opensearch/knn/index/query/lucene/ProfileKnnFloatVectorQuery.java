/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.*;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.profile.Timer;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;

public class ProfileKnnFloatVectorQuery extends KnnFloatVectorQuery {

    private QueryProfiler profiler;

    public ProfileKnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
        super(field, target, k, filter);
    }

    @Override
    public Query rewrite(IndexSearcher indexSearcher) throws IOException {
        profiler = ((ContextIndexSearcher) indexSearcher).getProfiler();
        return super.rewrite(indexSearcher);
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        Bits acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager
    ) throws IOException {
        if (profiler != null) {
            Timer timer = profiler.getProfileBreakdown(this).context(context).getTimer(KNNQueryTimingType.ANN_SEARCH);
            timer.start();
            try {
                return super.approximateSearch(context, acceptDocs, visitedLimit, knnCollectorManager);
            } finally {
                timer.stop();
            }
        }
        return super.approximateSearch(context, acceptDocs, visitedLimit, knnCollectorManager);
    }

    @Override
    protected TopDocs exactSearch(LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout)
        throws IOException {
        if (profiler != null) {
            Timer timer = profiler.getProfileBreakdown(this).context(context).getTimer(KNNQueryTimingType.EXACT_SEARCH);
            timer.start();
            try {
                return super.exactSearch(context, acceptIterator, queryTimeout);
            } finally {
                timer.stop();
            }
        }
        return super.exactSearch(context, acceptIterator, queryTimeout);
    }

}
