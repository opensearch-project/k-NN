/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;

/**
 * OpenSearch wrapper around Lucene's KnnByteVectorQuery that customizes
 * result merging to honor the original k parameter for queries with float vectors.
 *
 * <p>This wrapper ensures that when merging results from multiple segments, only the top k
 * documents are returned, maintaining consistency with OpenSearch's k-NN query behavior.
 */
public final class OSKnnFloatVectorQuery extends KnnFloatVectorQuery {
    private final int k;
    private final boolean needsRescore;

    public OSKnnFloatVectorQuery(
        final String fieldName,
        final float[] floatQueryVector,
        final int luceneK,
        final Query filterQuery,
        final int k,
        final boolean needsRescore
    ) {
        super(fieldName, floatQueryVector, luceneK, filterQuery);
        this.k = k;
        this.needsRescore = needsRescore;
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        if (needsRescore) {

            // When rescoring is enabled, we need to return all the oversampled k results to rescore which are later reduced to k
            return super.mergeLeafResults(perLeafResults);
        }
        // Merge all segment level results and take top k from it
        return TopDocs.merge(k, perLeafResults);
    }

}
