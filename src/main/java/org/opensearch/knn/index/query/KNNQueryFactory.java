/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.KnnVectorQuery;
import org.apache.lucene.search.Query;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Optional;

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
        final CreateQueryRequest createQueryRequest = CreateQueryRequest.builder()
            .knnEngine(knnEngine)
            .indexName(indexName)
            .fieldName(fieldName)
            .vector(vector)
            .k(k)
            .build();
        return create(createQueryRequest);
    }

    /**
     * Creates a Lucene query for a particular engine.
     * @param createQueryRequest request object that has all required fields to construct the query
     * @return Lucene Query
     */
    public static Query create(CreateQueryRequest createQueryRequest) {
        // Engines that create their own custom segment files cannot use the Lucene's KnnVectorQuery. They need to
        // use the custom query type created by the plugin
        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(createQueryRequest.getKnnEngine())) {
            log.debug(
                String.format(
                    "Creating custom k-NN query for index: %s \"\", field: %s \"\", k: %d",
                    createQueryRequest.getIndexName(),
                    createQueryRequest.getFieldName(),
                    createQueryRequest.getK()
                )
            );
            return new KNNQuery(
                createQueryRequest.getFieldName(),
                createQueryRequest.getVector(),
                createQueryRequest.getK(),
                createQueryRequest.getIndexName()
            );
        }

        log.debug(
            String.format(
                "Creating Lucene k-NN query for index: %s \"\", field: %s \"\", k: %d",
                createQueryRequest.getIndexName(),
                createQueryRequest.getFieldName(),
                createQueryRequest.getK()
            )
        );
        if (createQueryRequest.getFilter().isPresent()) {
            final QueryShardContext queryShardContext = createQueryRequest.getContext()
                .orElseThrow(() -> new RuntimeException("Shard context cannot be null"));
            try {
                final Query filterQuery = createQueryRequest.getFilter().get().toQuery(queryShardContext);
                return new KnnVectorQuery(
                    createQueryRequest.getFieldName(),
                    createQueryRequest.getVector(),
                    createQueryRequest.getK(),
                    filterQuery
                );
            } catch (IOException e) {
                throw new RuntimeException("Cannot create knn query with filter", e);
            }
        }
        return new KnnVectorQuery(createQueryRequest.getFieldName(), createQueryRequest.getVector(), createQueryRequest.getK());
    }

    /**
     * DTO object to hold data required to create a Query instance.
     */
    @AllArgsConstructor
    @Builder
    @Setter
    static class CreateQueryRequest {
        @Getter
        @NonNull
        private KNNEngine knnEngine;
        @Getter
        @NonNull
        private String indexName;
        @Getter
        private String fieldName;
        @Getter
        private float[] vector;
        @Getter
        private int k;
        // can be null in cases filter not passed with the knn query
        private QueryBuilder filter;
        // can be null in cases filter not passed with the knn query
        private QueryShardContext context;

        public Optional<QueryBuilder> getFilter() {
            return Optional.ofNullable(filter);
        }

        public Optional<QueryShardContext> getContext() {
            return Optional.ofNullable(context);
        }
    }
}
