/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;

/**
 * OpenSearch wrapper around Lucene's KnnByteVectorQuery that customizes
 * result merging to honor the original k parameter for queries with byte vectors.
 *
 * <p>This wrapper ensures that when merging results from multiple segments, only the top k
 * documents are returned, maintaining consistency with OpenSearch's k-NN query behavior.
 */
public final class OSKnnByteVectorQuery extends KnnByteVectorQuery {
    private final int k;

    public OSKnnByteVectorQuery(
        final String fieldName,
        final byte[] byteQueryVector,
        final int luceneK,
        final Query filterQuery,
        final int k
    ) {
        super(fieldName, byteQueryVector, luceneK, filterQuery);
        this.k = k;
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        // Merge all segment level results and take top k from it
        return TopDocs.merge(k, perLeafResults);
    }

}
