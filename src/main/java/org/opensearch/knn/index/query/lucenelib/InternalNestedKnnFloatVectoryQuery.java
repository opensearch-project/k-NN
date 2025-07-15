/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.opensearch.knn.profile.query.KNNMetrics;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.internal.ContextIndexSearcher;
import org.opensearch.search.profile.Profilers;
import org.opensearch.search.profile.Timer;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;

/**
 * InternalNestedKnnVectorQuery for float vector
 */
@Getter
public class InternalNestedKnnFloatVectoryQuery extends KnnFloatVectorQuery implements InternalNestedKnnVectorQuery {
    private final String field;
    private final float[] target;
    private final Query filter;
    private final int k;
    private final BitSetProducer parentFilter;
    private final DiversifyingChildrenFloatKnnVectorQuery diversifyingChildrenFloatKnnVectorQuery;

    private QueryProfiler profiler;

    public InternalNestedKnnFloatVectoryQuery(
        final String field,
        final float[] target,
        final Query filter,
        final int k,
        final BitSetProducer parentFilter
    ) {
        super(field, target, Integer.MAX_VALUE, filter);
        this.field = field;
        this.target = target;
        this.filter = filter;
        this.k = k;
        this.parentFilter = parentFilter;
        this.diversifyingChildrenFloatKnnVectorQuery = new DiversifyingChildrenFloatKnnVectorQuery(field, target, filter, k, parentFilter);
    }

    @Override
    public Query knnRewrite(final IndexSearcher searcher) throws IOException {
        profiler = ((ContextIndexSearcher) searcher).getProfiler();
        return diversifyingChildrenFloatKnnVectorQuery.rewrite(searcher);
    }

    @Override
    public TopDocs knnExactSearch(LeafReaderContext context, DocIdSetIterator acceptIterator) throws IOException {
        if (profiler != null) {
            Timer timer = profiler.getProfileBreakdown(this).context(context).getTimer(KNNQueryTimingType.EXACT_SEARCH);
            timer.start();
            try {
                return super.exactSearch(context, acceptIterator, null);
            } finally {
                timer.stop();
            }
        }
        return super.exactSearch(context, acceptIterator, null);
    }
}
