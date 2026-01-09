/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;

/**
 * OpenSearch wrapper around Lucene's DiversifyingChildrenFloatKnnVectorQuery that customizes
 * result merging to honor the original k parameter for nested field queries with float vectors.
 *
 * <p>This wrapper ensures that when merging results from multiple segments, only the top k
 * documents are returned, maintaining consistency with OpenSearch's k-NN query behavior.
 */
public final class OSDiversifyingChildrenFloatKnnVectorQuery extends DiversifyingChildrenFloatKnnVectorQuery {
    private final int k;

    public OSDiversifyingChildrenFloatKnnVectorQuery(
        final String fieldName,
        final float[] vector,
        final Query filterQuery,
        final int luceneK,
        final BitSetProducer parentFilter,
        final int k
    ) {
        super(fieldName, vector, filterQuery, luceneK, parentFilter);
        this.k = k;
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        // Merge all segment level results and take top k from it
        return TopDocs.merge(k, perLeafResults);
    }
}
