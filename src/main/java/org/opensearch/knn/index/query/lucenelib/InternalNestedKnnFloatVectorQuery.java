/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import lombok.Getter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;

import java.io.IOException;

/**
 * InternalNestedKnnVectorQuery for float vector
 */
@Getter
public class InternalNestedKnnFloatVectorQuery extends KnnFloatVectorQuery implements InternalNestedKnnVectorQuery {
    private final String field;
    private final float[] target;
    private final Query filter;
    private final int luceneK; // Number of nearest neighbors to retrieve from Lucene (augmented k)
    private final BitSetProducer parentFilter;
    private final int k; // Number of nearest neighbors requested by the user query
    private final OSDiversifyingChildrenFloatKnnVectorQuery osDiversifyingChildrenFloatKnnVectorQuery;

    public InternalNestedKnnFloatVectorQuery(
        final String field,
        final float[] target,
        final Query filter,
        final int luceneK,
        final BitSetProducer parentFilter,
        final int k,
        final boolean needsRescore,
        final boolean expandNestedDocs
    ) {
        super(field, target, Integer.MAX_VALUE, filter);
        this.field = field;
        this.target = target;
        this.filter = filter;
        this.luceneK = luceneK;
        this.parentFilter = parentFilter;
        this.k = k;
        this.osDiversifyingChildrenFloatKnnVectorQuery = new OSDiversifyingChildrenFloatKnnVectorQuery(
            field,
            target,
            filter,
            luceneK,
            parentFilter,
            k,
            needsRescore,
            expandNestedDocs
        );
    }

    @Override
    public Query knnRewrite(final IndexSearcher searcher) throws IOException {
        return osDiversifyingChildrenFloatKnnVectorQuery.rewrite(searcher);
    }

    @Override
    public TopDocs knnExactSearch(LeafReaderContext context, DocIdSetIterator acceptIterator) throws IOException {
        return super.exactSearch(context, acceptIterator, null);
    }
}
