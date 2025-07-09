/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.profile.query.KNNMetrics;
import org.opensearch.knn.profile.query.LuceneEngineKnnTimingType;
import org.opensearch.search.profile.Profilers;
import org.opensearch.search.profile.Timer;

import java.io.IOException;

public class ProfileDiversifyingChildrenFloatKnnVectorQuery extends DiversifyingChildrenFloatKnnVectorQuery {

    private final Profilers profilers;

    public ProfileDiversifyingChildrenFloatKnnVectorQuery(
        String field,
        float[] target,
        Query childFilter,
        int k,
        BitSetProducer parentsFilter
    ) {
        super(field, target, childFilter, k, parentsFilter);
        profilers = KNNMetrics.getProfilers();
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        Bits acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager
    ) throws IOException {
        if (profilers != null) {
            Timer timer = profilers.getCurrentQueryProfiler()
                .getTopBreakdown()
                .context(context)
                .getTimer(LuceneEngineKnnTimingType.ANN_SEARCH);
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
        if (profilers != null) {
            Timer timer = profilers.getCurrentQueryProfiler()
                .getTopBreakdown()
                .context(context)
                .getTimer(LuceneEngineKnnTimingType.EXACT_SEARCH);
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
