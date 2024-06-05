/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.search.NestedHelper;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

/**
* Base class for creating vector search queries.
*/
@Log4j2
public abstract class BaseQueryFactory {
    /**
     * DTO object to hold data required to create a Query instance.
     */
    @AllArgsConstructor
    @Builder
    @Getter
    public static class CreateQueryRequest {
        @NonNull
        private KNNEngine knnEngine;
        @NonNull
        private String indexName;
        private String fieldName;
        private float[] vector;
        private byte[] byteVector;
        private VectorDataType vectorDataType;
        private Map<String, ?> methodParameters;
        private Integer k;
        private Float radius;
        private QueryBuilder filter;
        private QueryShardContext context;

        public Optional<QueryBuilder> getFilter() {
            return Optional.ofNullable(filter);
        }

        public Optional<QueryShardContext> getContext() {
            return Optional.ofNullable(context);
        }
    }

    /**
     * Creates a query filter.
     *
     * @param createQueryRequest request object that has all required fields to construct the query
     * @return Lucene Query
     */
    protected static Query getFilterQuery(BaseQueryFactory.CreateQueryRequest createQueryRequest) {
        if (!createQueryRequest.getFilter().isPresent()) {
            return null;
        }

        final QueryShardContext queryShardContext = createQueryRequest.getContext()
            .orElseThrow(() -> new RuntimeException("Shard context cannot be null"));
        log.debug(
            String.format(
                "Creating query with filter for index [%s], field [%s]",
                createQueryRequest.getIndexName(),
                createQueryRequest.getFieldName()
            )
        );
        final Query filterQuery;
        try {
            filterQuery = createQueryRequest.getFilter().get().toQuery(queryShardContext);
        } catch (IOException e) {
            throw new RuntimeException("Cannot create query with filter", e);
        }
        BitSetProducer parentFilter = queryShardContext.getParentFilter();
        if (parentFilter != null) {
            boolean mightMatch = new NestedHelper(queryShardContext.getMapperService()).mightMatchNestedDocs(filterQuery);
            if (mightMatch) {
                return filterQuery;
            }
            return new ToChildBlockJoinQuery(filterQuery, parentFilter);
        }
        return filterQuery;
    }
}
