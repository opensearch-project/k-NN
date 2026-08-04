/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;

/**
 * OpenSearch wrapper around Lucene's DiversifyingChildrenFloatKnnVectorQuery that customizes
 * result merging to honor the original k parameter for nested field queries with float vectors.
 *
 * <p>This wrapper ensures that when merging results from multiple segments, only the top k
 * documents are returned, maintaining consistency with OpenSearch's k-NN query behavior.
 */
public final class OSDiversifyingChildrenFloatKnnVectorQuery extends DiversifyingChildrenFloatKnnVectorQuery {

    private final int k;
    private final int rescoreK;
    private final boolean expandNestedDocs;
    private final BitSetProducer parentFilter;

    public OSDiversifyingChildrenFloatKnnVectorQuery(
        final String fieldName,
        final float[] vector,
        final Query filterQuery,
        final int luceneK,
        final BitSetProducer parentFilter,
        final int k,
        final int rescoreK
    ) {
        this(fieldName, vector, filterQuery, luceneK, parentFilter, k, rescoreK, false);
    }

    public OSDiversifyingChildrenFloatKnnVectorQuery(
        final String fieldName,
        final float[] vector,
        final Query filterQuery,
        final int luceneK,
        final BitSetProducer parentFilter,
        final int k,
        final int rescoreK,
        final boolean expandNestedDocs
    ) {
        super(fieldName, vector, filterQuery, luceneK, parentFilter);
        this.k = k;
        this.rescoreK = rescoreK;
        this.expandNestedDocs = expandNestedDocs;
        this.parentFilter = parentFilter;
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager
    ) throws IOException {
        if (NestedKnnUtil.hasNoParentDocs(parentFilter, context)) {
            return NestedKnnUtil.EMPTY_TOP_DOCS;
        }
        return super.approximateSearch(context, acceptDocs, visitedLimit, knnCollectorManager);
    }

    /**
     * Performs a diversified, full-precision exact search over the accepted child documents, returning the
     * best-scoring child per parent. Reads raw float vectors, so it is full precision even for quantized
     * (e.g. 4x / on_disk) indexes. Used by {@link ExpandNestedDocsQuery} to rescore the oversampled parent
     * candidates before expanding their nested documents.
     *
     * @param context the leaf reader context
     * @param acceptIterator iterator over the candidate child documents
     * @return diversified top docs (best child per parent)
     * @throws IOException on read error
     */
    public TopDocs diversifyingExactSearch(final LeafReaderContext context, final DocIdSetIterator acceptIterator) throws IOException {
        return super.exactSearch(context, acceptIterator, null);
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        if (rescoreK != RescoreContext.NO_RESCORE_NEEDED) {
            // When rescoring is enabled, merge to the oversampled k (rescore budget) rather than the full
            // luceneK which may have been expanded by ef_search. For the expandNested path this preserves the
            // oversampled parent candidates so ExpandNestedDocsQuery can rescore them at full precision and
            // reduce to the top k parents before expanding their child documents.
            return TopDocs.merge(rescoreK, perLeafResults);
        }
        // Merge all segment level results and take top k from it
        return TopDocs.merge(k, perLeafResults);
    }
}
