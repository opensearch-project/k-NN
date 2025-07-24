/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.*;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.profile.KNNProfileUtil;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.profile.query.QueryProfiler;

import java.io.IOException;

/**
 * Wrapper class used for profiling {@link DiversifyingChildrenFloatKnnVectorQuery}
 */
public class ProfileDiversifyingChildrenFloatKnnVectorQuery extends DiversifyingChildrenFloatKnnVectorQuery {

    private QueryProfiler profiler;

    public ProfileDiversifyingChildrenFloatKnnVectorQuery(
        String field,
        float[] target,
        Query childFilter,
        int k,
        BitSetProducer parentsFilter
    ) {
        super(field, target, childFilter, k, parentsFilter);
    }

    @Override
    public Query rewrite(IndexSearcher indexSearcher) throws IOException {
        profiler = KNNProfileUtil.getProfiler(indexSearcher);
        return super.rewrite(indexSearcher);
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        Bits acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager
    ) throws IOException {
        return (TopDocs) KNNProfileUtil.profile(profiler, this, context, KNNQueryTimingType.ANN_SEARCH, () -> {
            try {
                return super.approximateSearch(context, acceptDocs, visitedLimit, knnCollectorManager);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Override
    protected TopDocs exactSearch(LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout)
        throws IOException {
        return (TopDocs) KNNProfileUtil.profile(profiler, this, context, KNNQueryTimingType.EXACT_SEARCH, () -> {
            try {
                return super.exactSearch(context, acceptIterator, queryTimeout);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }
}
