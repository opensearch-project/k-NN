/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.KnnVectorQuery;
import org.apache.lucene.search.Query;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;

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
    public static Query create(KNNEngine knnEngine, String indexName, String fieldName, float[] vector, int k, KNNQueryFilter knnQueryFilter, QueryShardContext context) {
        // Engines that create their own custom segment files cannot use the Lucene's KnnVectorQuery. They need to
        // use the custom query type created by the plugin
        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine)) {
            log.debug(String.format("Creating custom k-NN query for index: %s \"\", field: %s \"\", k: %d", indexName, fieldName, k));
            return new KNNQuery(fieldName, vector, k, indexName);
        }

        log.debug(String.format("Creating Lucene k-NN query for index: %s \"\", field: %s \"\", k: %d", indexName, fieldName, k));
        if (knnQueryFilter == null) {
            return new KnnVectorQuery(fieldName, vector, k);
        }
        BooleanQuery.Builder booleanQuery;
        boolean filterAdded = false;
        try {
            booleanQuery = new BooleanQuery.Builder();
            if (!knnQueryFilter.getMustClauses().isEmpty()) {
                for (QueryBuilder mustQuery : knnQueryFilter.getMustClauses()) {
                    booleanQuery.add(mustQuery.toQuery(context), BooleanClause.Occur.MUST);
                }
                filterAdded = true;
            }
            if (!knnQueryFilter.getShouldClauses().isEmpty()) {
                for (QueryBuilder shouldQuery : knnQueryFilter.getShouldClauses()) {
                    booleanQuery.add(shouldQuery.toQuery(context), BooleanClause.Occur.SHOULD);
                }
                filterAdded = true;
            }
            if (!knnQueryFilter.getMustNotClauses().isEmpty()) {
                for (QueryBuilder mustNotQuery : knnQueryFilter.getMustNotClauses()) {
                    booleanQuery.add(mustNotQuery.toQuery(context), BooleanClause.Occur.MUST_NOT);
                }
                filterAdded = true;
            }
        } catch (IOException e) {
            throw new RuntimeException("Cannot construct filter for knn query", e);
        }
        return filterAdded ? new KnnVectorQuery(fieldName, vector, k, booleanQuery.build()) : new KnnVectorQuery(fieldName, vector, k);
    }
}
