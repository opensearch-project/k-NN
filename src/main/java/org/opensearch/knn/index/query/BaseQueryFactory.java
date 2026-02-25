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
import org.opensearch.common.lucene.search.Queries;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.search.NestedHelper;
import org.opensearch.index.search.OpenSearchToParentBlockJoinQuery;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;
import java.util.Deque;
import java.util.LinkedList;
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
        private float[] originalVector;
        private byte[] byteVector;
        private VectorDataType vectorDataType;
        private Map<String, ?> methodParameters;
        private Integer k;
        private Float radius;
        private QueryBuilder filter;
        private QueryShardContext context;
        private RescoreContext rescoreContext;
        private boolean expandNested;
        private boolean memoryOptimizedSearchEnabled;

        public Optional<QueryBuilder> getFilter() {
            return Optional.ofNullable(filter);
        }

        public Optional<QueryShardContext> getContext() {
            return Optional.ofNullable(context);
        }

        public Optional<RescoreContext> getRescoreContext() {
            return Optional.ofNullable(rescoreContext);
        }

        public boolean isExpandNested() {
            return expandNested;
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

        // We want to evaluate the filter node's query at the root level.
        // Unwind and preserve the nested object stack.
        Deque<ObjectMapper> nestedLevelStack = new LinkedList<>();
        ObjectMapper objectMapper = null;
        if (queryShardContext.nestedScope() != null) {
            while ((objectMapper = queryShardContext.nestedScope().getObjectMapper()) != null) {
                nestedLevelStack.push(objectMapper);
                queryShardContext.nestedScope().previousLevel();
            }
        }

        final Query filterQuery;
        try {
            filterQuery = createQueryRequest.getFilter().get().toQuery(queryShardContext);
        } catch (IOException e) {
            throw new RuntimeException("Cannot create query with filter", e);
        } finally {
            // Rewind the nested object stack before returning.
            while ((objectMapper = nestedLevelStack.peek()) != null) {
                queryShardContext.nestedScope().nextLevel(objectMapper);
                nestedLevelStack.pop();
            }
        }

        if (filterQuery != null && queryShardContext.getParentFilter() != null) {
            // This k-NN query is executing in nested context. Query nodes beneath nested
            // queries must match child documents. However, the efficient filter query in
            // the k-NN API is designed to work with root-level parent documents. Joining
            // down to child documents is therefore required to make this work.

            // To-parent joins which target the root level can simply be unwrapped to get
            // a query matching the desired child documents.
            if (filterQuery instanceof OpenSearchToParentBlockJoinQuery) {
                final OpenSearchToParentBlockJoinQuery toParentQuery = (OpenSearchToParentBlockJoinQuery) filterQuery;

                if (toParentQuery.getPath() == null) {
                    return toParentQuery.getChildQuery();
                }
            }

            // Set up for performing the join.
            final Query parentQuery = Queries.newNonNestedFilter();
            final BitSetProducer parentFilter = queryShardContext.bitsetFilter(parentQuery);

            // mightMatchNestedDocs() is conservative and only returns false if the query
            // can never match a nested document. If it returns true, a filter to exclude
            // nested documents should be applied first to guarantee they never match.
            final Query nonNestedFilterQuery;

            if (new NestedHelper(queryShardContext.getMapperService()).mightMatchNestedDocs(filterQuery)) {
                nonNestedFilterQuery = Queries.filtered(filterQuery, parentQuery);
            } else {
                nonNestedFilterQuery = filterQuery;
            }

            return new ToChildBlockJoinQuery(nonNestedFilterQuery, parentFilter);
        }

        return filterQuery;
    }
}
