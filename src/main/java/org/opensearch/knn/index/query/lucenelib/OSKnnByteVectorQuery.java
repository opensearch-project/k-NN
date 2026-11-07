/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.opensearch.knn.index.query.rescore.RescoreContext;

/**
 * OpenSearch wrapper around Lucene's KnnByteVectorQuery that customizes
 * result merging to honor the original k parameter for queries with byte vectors.
 *
 * <p>This wrapper ensures that when merging results from multiple segments, only the top k
 * documents are returned, maintaining consistency with OpenSearch's k-NN query behavior.
 */
public final class OSKnnByteVectorQuery extends KnnByteVectorQuery {
    private final int k;
    private final int rescoreK;

    public OSKnnByteVectorQuery(
        final String fieldName,
        final byte[] byteQueryVector,
        final int luceneK,
        final Query filterQuery,
        final int k
    ) {
        this(fieldName, byteQueryVector, luceneK, filterQuery, k, RescoreContext.NO_RESCORE_NEEDED);
    }

    public OSKnnByteVectorQuery(
        final String fieldName,
        final byte[] byteQueryVector,
        final int luceneK,
        final Query filterQuery,
        final int k,
        final int rescoreK
    ) {
        super(fieldName, byteQueryVector, luceneK, filterQuery);
        this.k = k;
        this.rescoreK = rescoreK;
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        if (rescoreK != RescoreContext.NO_RESCORE_NEEDED) {
            // When rescoring is enabled, merge to oversampled k (rescore budget) rather than the
            // full luceneK which may have been expanded by ef_search.
            return TopDocs.merge(rescoreK, perLeafResults);
        }
        // Merge all segment level results and take top k from it
        return TopDocs.merge(k, perLeafResults);
    }

}
