/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.index.query.common.QueryUtils;

/**
 * A class to create a nested knn vector query for lucene
 */
public class NestedKnnVectorQueryFactory {
    /**
     * Create a query for k-NN nested field.
     *
     * The query is generated two times when inner_hits() parameter exist in the request.
     * For inner hit, we return all filtered nested field documents belongs to the final result of parent documents.
     *
     * @param fieldName field name for search
     * @param vector target vector for search
     * @param luceneK k value used for Lucene search, augmented based on efSearch
     * @param filterQuery efficient filtering query
     * @param parentFilter has mapping data between parent doc and child doc
     * @param expandNestedDocs tells if this query is for expanding nested docs
     * @param k k nearest neighbor for search
     * @return Query for k-NN nested field
     */
    public static Query createNestedKnnVectorQuery(
        final String fieldName,
        final byte[] vector,
        final int luceneK,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNestedDocs,
        final int k
    ) {
        if (expandNestedDocs) {
            return new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(
                new InternalNestedKnnByteVectorQuery(fieldName, vector, filterQuery, luceneK, parentFilter, k)
            ).queryUtils(QueryUtils.getInstance()).build();
        }
        return new OSDiversifyingChildrenByteKnnVectorQuery(fieldName, vector, filterQuery, luceneK, parentFilter, k);
    }

    /**
     * Create a query for k-NN nested field.
     *
     * The query is generated two times when inner_hits() parameter exist in the request.
     * For inner hit, we return all filtered nested field documents belongs to the final result of parent documents.
     *
     * @param fieldName field name for search
     * @param vector target vector for search
     * @param luceneK k value used for Lucene search, augmented based on efSearch
     * @param filterQuery efficient filtering query
     * @param parentFilter has mapping data between parent doc and child doc
     * @param expandNestedDocs tells if this query is for expanding nested docs
     * @param k k nearest neighbor for search
     * @return Query for k-NN nested field
     */
    public static Query createNestedKnnVectorQuery(
        final String fieldName,
        final float[] vector,
        final int luceneK,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNestedDocs,
        final int k
    ) {
        if (expandNestedDocs) {
            return new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(
                new InternalNestedKnnFloatVectorQuery(fieldName, vector, filterQuery, luceneK, parentFilter, k)
            ).queryUtils(QueryUtils.getInstance()).build();
        }
        return new OSDiversifyingChildrenFloatKnnVectorQuery(fieldName, vector, filterQuery, luceneK, parentFilter, k);
    }
}
