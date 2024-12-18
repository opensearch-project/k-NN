/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenByteKnnVectorQuery;

import java.io.IOException;

/**
 * InternalNestedKnnVectorQuery for byte vector
 */
@Getter
public class InternalNestedKnnByteVectoryQuery extends KnnByteVectorQuery implements InternalNestedKnnVectorQuery {
    private final String field;
    private final byte[] target;
    private final Query filter;
    private final int k;
    private final BitSetProducer parentFilter;
    private final DiversifyingChildrenByteKnnVectorQuery diversifyingChildrenByteKnnVectorQuery;

    public InternalNestedKnnByteVectoryQuery(
        final String field,
        final byte[] target,
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
        this.diversifyingChildrenByteKnnVectorQuery = new DiversifyingChildrenByteKnnVectorQuery(field, target, filter, k, parentFilter);
    }

    @Override
    public Query knnRewrite(final IndexSearcher searcher) throws IOException {
        return diversifyingChildrenByteKnnVectorQuery.rewrite(searcher);
    }

    @Override
    public TopDocs knnExactSearch(LeafReaderContext context, DocIdSetIterator acceptIterator) throws IOException {
        return super.exactSearch(context, acceptIterator, null);
    }
}
