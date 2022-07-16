/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.KnnVectorQuery;
import org.apache.lucene.search.Query;
import org.opensearch.knn.index.util.KNNEngine;

/**
 * Creates the Lucene k-NN queries
 */
public class KNNQueryFactory {

    /**
     * Creates a Lucene query for a particular engine.
     *
     * @param knnEngine Engine to create the query for
     * @param indexName Name of the OpenSearch index that is being queried
     * @param fieldName Name of the field in the OpenSearch index that will be queried
     * @param vector The query vector to get the nearest neighbors for
     * @param k the number of nearest neighbors to return
     * @return Lucene Query
     */
    public static Query create(KNNEngine knnEngine, String indexName, String fieldName, float[] vector, int k) {
        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine)) {
            return new CustomKNNQuery(fieldName, vector, k, indexName);
        }
        
        return new KnnVectorQuery(fieldName, vector, k);
    }
}
