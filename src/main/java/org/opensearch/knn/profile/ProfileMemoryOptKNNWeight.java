/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight;
import org.opensearch.knn.profile.query.KNNMetrics;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.profile.ContextualProfileBreakdown;

import java.io.IOException;

/**
 * Wrapper class used when profiling {@link org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight}
 */
public class ProfileMemoryOptKNNWeight extends MemoryOptimizedKNNWeight {

    protected final ContextualProfileBreakdown profile;

    /**
     * Wrapper constructor around {@link MemoryOptimizedKNNWeight}'s constructor used to
     * keep track of the {@link KNNQuery}'s profile breakdown
     * @param query
     * @param boost
     * @param filterWeight
     * @param searcher
     * @param k
     * @param profile ContextualProfileBreakdown based on KNNQuery
     */
    public ProfileMemoryOptKNNWeight(
        KNNQuery query,
        float boost,
        final Weight filterWeight,
        IndexSearcher searcher,
        int k,
        ContextualProfileBreakdown profile
    ) {
        super(query, boost, filterWeight, searcher, k);
        this.profile = profile;
    }

    @Override
    protected BitSet createBitSet(
        final LeafReaderContext ctx,
        final DocIdSetIterator filteredDocIdsIterator,
        final Bits liveDocs,
        int maxDoc
    ) throws IOException {
        BitSet filterBitSet = (BitSet) KNNProfileUtil.profileBreakdown(
            profile,
            ctx,
            KNNQueryTimingType.BITSET_CREATION,
            () -> super.createBitSet(ctx, filteredDocIdsIterator, liveDocs, maxDoc)
        );
        LongMetric cardMetric = (LongMetric) profile.context(ctx).getMetric(KNNMetrics.CARDINALITY);
        cardMetric.setValue((long) filterBitSet.cardinality());
        return filterBitSet;
    }

    @Override
    public TopDocs approximateSearch(final LeafReaderContext context, final BitSet filterIdsBitSet, final int cardinality, final int k)
        throws IOException {
        return (TopDocs) KNNProfileUtil.profileBreakdown(
            profile,
            context,
            KNNQueryTimingType.ANN_SEARCH,
            () -> super.approximateSearch(context, filterIdsBitSet, cardinality, k)
        );
    }

    @Override
    public TopDocs exactSearch(final LeafReaderContext leafReaderContext, final ExactSearcher.ExactSearcherContext exactSearcherContext)
        throws IOException {
        // Set cardinality from exactSearcherContext - this handles both cases:
        // 1. When BitSet was created: cardinality was already set, this overwrites with same value
        // 2. When BitSet was NOT created (iterator optimization): this sets the cardinality
        LongMetric cardMetric = (LongMetric) profile.context(leafReaderContext).getMetric(KNNMetrics.CARDINALITY);
        cardMetric.setValue(exactSearcherContext.getNumberOfMatchedDocs());
        return (TopDocs) KNNProfileUtil.profileBreakdown(
            profile,
            leafReaderContext,
            KNNQueryTimingType.EXACT_SEARCH,
            () -> super.exactSearch(leafReaderContext, exactSearcherContext)
        );
    }
}
