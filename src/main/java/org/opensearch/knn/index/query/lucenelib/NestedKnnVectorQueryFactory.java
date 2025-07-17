/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.lucene.ProfileDiversifyingChildrenByteKnnVectorQuery;
import org.opensearch.knn.index.query.lucene.ProfileDiversifyingChildrenFloatKnnVectorQuery;

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
     * @param k k nearest neighbor for search
     * @param filterQuery efficient filtering query
     * @param parentFilter has mapping data between parent doc and child doc
     * @param expandNestedDocs tells if this query is for expanding nested docs
     * @return Query for k-NN nested field
     */
    public static Query createNestedKnnVectorQuery(
        final String fieldName,
        final byte[] vector,
        final int k,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNestedDocs
    ) {
        if (expandNestedDocs) {
            return new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(
                new InternalNestedKnnByteVectoryQuery(fieldName, vector, filterQuery, k, parentFilter)
            ).queryUtils(QueryUtils.getInstance()).build();
        }
        return new ProfileDiversifyingChildrenByteKnnVectorQuery(fieldName, vector, filterQuery, k, parentFilter);
    }

    /**
     * Create a query for k-NN nested field.
     *
     * The query is generated two times when inner_hits() parameter exist in the request.
     * For inner hit, we return all filtered nested field documents belongs to the final result of parent documents.
     *
     * @param fieldName field name for search
     * @param vector target vector for search
     * @param k k nearest neighbor for search
     * @param filterQuery efficient filtering query
     * @param parentFilter has mapping data between parent doc and child doc
     * @param expandNestedDocs tells if this query is for expanding nested docs
     * @return Query for k-NN nested field
     */
    public static Query createNestedKnnVectorQuery(
        final String fieldName,
        final float[] vector,
        final int k,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNestedDocs
    ) {
        if (expandNestedDocs) {
            return new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(
                new InternalNestedKnnFloatVectoryQuery(fieldName, vector, filterQuery, k, parentFilter)
            ).queryUtils(QueryUtils.getInstance()).build();
        }
        return new ProfileDiversifyingChildrenFloatKnnVectorQuery(fieldName, vector, filterQuery, k, parentFilter);
    }
}
