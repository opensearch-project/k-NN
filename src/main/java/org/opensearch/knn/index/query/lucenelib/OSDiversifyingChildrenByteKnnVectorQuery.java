/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenByteKnnVectorQuery;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.util.BitSet;

import java.io.IOException;

/**
 * OpenSearch wrapper around Lucene's DiversifyingChildrenByteKnnVectorQuery that customizes
 * result merging to honor the original k parameter for nested field queries with byte vectors.
 *
 * <p>This wrapper ensures that when merging results from multiple segments, only the top k
 * documents are returned, maintaining consistency with OpenSearch's k-NN query behavior.
 */
public final class OSDiversifyingChildrenByteKnnVectorQuery extends DiversifyingChildrenByteKnnVectorQuery {
    private final int k;
    private final BitSetProducer parentFilter;

    public OSDiversifyingChildrenByteKnnVectorQuery(
        final String fieldName,
        final byte[] vector,
        final Query filterQuery,
        final int luceneK,
        final BitSetProducer parentFilter,
        final int k
    ) {
        super(fieldName, vector, filterQuery, luceneK, parentFilter);
        this.k = k;
        this.parentFilter = parentFilter;
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager
    ) throws IOException {
        // Check if the segment has parent documents (nested docs). If not, return empty results.
        // This prevents a NPE in Lucene's TimeLimitingKnnCollectorManager which wraps a null
        // collector when DiversifyingNearestChildrenKnnCollectorManager returns null (no parent
        // BitSet in segment), then topDocs() is called on the null inner collector.
        BitSet parentBitSet = parentFilter.getBitSet(context);
        if (parentBitSet == null) {
            return new TopDocs(
                new org.apache.lucene.search.TotalHits(0, org.apache.lucene.search.TotalHits.Relation.EQUAL_TO),
                new org.apache.lucene.search.ScoreDoc[0]
            );
        }
        return super.approximateSearch(context, acceptDocs, visitedLimit, knnCollectorManager);
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        // Merge all segment level results and take top k from it
        return TopDocs.merge(k, perLeafResults);
    }
}
