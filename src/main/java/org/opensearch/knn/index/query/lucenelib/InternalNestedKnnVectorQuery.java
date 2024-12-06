/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;

import java.io.IOException;

/**
 * Query interface to support k-NN nested field
 */
public interface InternalNestedKnnVectorQuery {
    /**
     * Return a rewritten query of nested knn search
     *
     * @param searcher index searcher
     * @return rewritten query of nested knn search
     * @throws IOException
     */
    Query knnRewrite(final IndexSearcher searcher) throws IOException;

    /**
     * Return a result of exact knn search
     *
     * @param leafReaderContext segment context
     * @param iterator filtered doc ids
     * @return
     * @throws IOException
     */
    TopDocs knnExactSearch(final LeafReaderContext leafReaderContext, final DocIdSetIterator iterator) throws IOException;

    /**
     * Return a field name
     * @return field name
     */
    String getField();

    /**
     * Return a filter query
     * @return filter query
     */
    Query getFilter();

    /**
     * Return k value
     * @return k value
     */
    int getK();

    /**
     * Return a parent filter
     * @return parent filter
     */
    BitSetProducer getParentFilter();
}
