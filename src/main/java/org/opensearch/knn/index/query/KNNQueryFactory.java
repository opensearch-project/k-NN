/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.KnnVectorQuery;
import org.apache.lucene.search.Query;
import org.opensearch.knn.index.util.KNNEngine;

/**
 * Creates the Lucene k-NN queries
 */
@Log4j2
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
        // Engines that create their own custom segment files cannot use the Lucene's KnnVectorQuery. They need to
        // use the custom query type created by the plugin
        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine)) {
            log.debug(String.format("Creating custom k-NN query for index: %s \"\", field: %s \"\", k: %d", indexName, fieldName, k));
            return new KNNQuery(fieldName, vector, k, indexName);
        }

        log.debug(String.format("Creating Lucene k-NN query for index: %s \"\", field: %s \"\", k: %d", indexName, fieldName, k));
        return new KnnVectorQuery(fieldName, vector, k);
    }
}
