/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;

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
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        // TODO: Fix this if condition after adding rescoring logic inside ExpandNestedDocsQuery when rescoring is enabled
        if (rescoreK != RescoreContext.NO_RESCORE_NEEDED && !expandNestedDocs) {
            // When rescoring is enabled, merge to oversampled k (rescore budget) rather than the
            // full luceneK which may have been expanded by ef_search.
            return TopDocs.merge(rescoreK, perLeafResults);
        }
        // Merge all segment level results and take top k from it
        return TopDocs.merge(k, perLeafResults);
    }
}
