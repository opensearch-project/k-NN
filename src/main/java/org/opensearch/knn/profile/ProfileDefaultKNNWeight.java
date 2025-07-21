/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.query.DefaultKNNWeight;
import org.opensearch.knn.index.query.ExactSearcher;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.profile.query.KNNMetrics;
import org.opensearch.knn.profile.query.KNNQueryTimingType;
import org.opensearch.search.profile.ContextualProfileBreakdown;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

/**
 * Wrapper class used when profiling {@link org.opensearch.knn.index.query.DefaultKNNWeight}
 */
public class ProfileDefaultKNNWeight extends DefaultKNNWeight {

    protected final ContextualProfileBreakdown profile;

    public ProfileDefaultKNNWeight(KNNQuery query, float boost, Weight filterWeight, ContextualProfileBreakdown profile) {
        super(query, boost, filterWeight);
        this.profile = profile;
    }

    @Override
    protected BitSet getFilteredDocsBitSet(final LeafReaderContext ctx) throws IOException {
        BitSet filterBitSet = (BitSet) KNNProfileUtil.profileBreakdown(
            profile,
            ctx,
            KNNQueryTimingType.BITSET_CREATION,
            () -> super.getFilteredDocsBitSet(ctx)
        );
        LongMetric cardMetric = (LongMetric) profile.context(ctx).getMetric(KNNMetrics.CARDINALITY);
        cardMetric.setValue((long) filterBitSet.cardinality());
        return filterBitSet;
    }

    @Override
    protected TopDocs approximateSearch(final LeafReaderContext context, final BitSet filterIdsBitSet, final int cardinality, final int k)
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
        return (TopDocs) KNNProfileUtil.profileBreakdown(
            profile,
            leafReaderContext,
            KNNQueryTimingType.EXACT_SEARCH,
            () -> super.exactSearch(leafReaderContext, exactSearcherContext)
        );
    }

    @Override
    protected NativeMemoryAllocation loadGraph(
        final SegmentReader reader,
        String cacheKey,
        final SpaceType spaceType,
        final KNNEngine knnEngine,
        final KNNQuery knnQuery,
        final VectorDataType vectorDataType,
        final byte[] quantizedVector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo,
        final String modelId,
        LeafReaderContext context
    ) throws ExecutionException, IOException {
        return (NativeMemoryAllocation) KNNProfileUtil.profileBreakdown(profile, context, KNNQueryTimingType.GRAPH_LOAD, () -> {
            try {
                return super.loadGraph(
                    reader,
                    cacheKey,
                    spaceType,
                    knnEngine,
                    knnQuery,
                    vectorDataType,
                    quantizedVector,
                    segmentLevelQuantizationInfo,
                    modelId,
                    context
                );
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            }
        });
    }
}
