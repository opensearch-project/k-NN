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
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.search.NestedHelper;
import org.opensearch.index.search.OpenSearchToParentBlockJoinQuery;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
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
        private RescoreContext rescoreContext;
        private Boolean expandNested;
        private boolean memoryOptimizedSearchSupported;

        public Optional<QueryBuilder> getFilter() {
            return Optional.ofNullable(filter);
        }

        public Optional<QueryShardContext> getContext() {
            return Optional.ofNullable(context);
        }

        public Optional<RescoreContext> getRescoreContext() {
            return Optional.ofNullable(rescoreContext);
        }

        public Optional<Boolean> getExpandNested() {
            return Optional.ofNullable(expandNested);
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

        //preserve nestedStack
        Deque<ObjectMapper> nestedLevelStack = new LinkedList<>();
        ObjectMapper objectMapper = null;
        while((objectMapper = queryShardContext.nestedScope().getObjectMapper()) != null) {
            nestedLevelStack.push(objectMapper);
            queryShardContext.nestedScope().previousLevel();
        }

        final Query filterQuery;
        try {
            filterQuery = createQueryRequest.getFilter().get().toQuery(queryShardContext);
        } catch (IOException e) {
            throw new RuntimeException("Cannot create query with filter", e);
        } finally {
            while((objectMapper = nestedLevelStack.peek()) != null) {
                queryShardContext.nestedScope().nextLevel(objectMapper);
                nestedLevelStack.pop();
            }
        }
        BitSetProducer parentFilter = queryShardContext.getParentFilter();
        if (parentFilter != null) {
            boolean mightMatch = new NestedHelper(queryShardContext.getMapperService()).mightMatchNestedDocs(filterQuery);
            if (mightMatch) {
                return filterQuery;
            } else if (filterQuery instanceof OpenSearchToParentBlockJoinQuery) {
                //this case would happen when path = null, and filter is nested
                return ((OpenSearchToParentBlockJoinQuery)filterQuery).getChildQuery();
            } else if (filterQuery instanceof BooleanQuery) {
                KNNQueryVisitor knnQueryVisitor = new KNNQueryVisitor();
                filterQuery.visit(knnQueryVisitor);
                BooleanQuery.Builder builder = (new BooleanQuery.Builder())
                        .add(new ToChildBlockJoinQuery(filterQuery, parentFilter), BooleanClause.Occur.FILTER);
                for(Query q : knnQueryVisitor.nestedQuery) {
                    builder.add(q, BooleanClause.Occur.FILTER);
                }
                return builder.build();
            }
            return new ToChildBlockJoinQuery(filterQuery, parentFilter);
        }
        return filterQuery;
    }

    @Getter
    static class KNNQueryVisitor extends QueryVisitor {
        List<Query> nestedQuery;
        public KNNQueryVisitor() {
            nestedQuery = new ArrayList<>();
        }
        public QueryVisitor getSubVisitor(BooleanClause.Occur occur, Query parent) {
            if (parent instanceof BooleanQuery && occur == BooleanClause.Occur.FILTER) {
                Collection<Query> collection = ((BooleanQuery)parent).getClauses(BooleanClause.Occur.FILTER);
                for(Query q : collection) {
                    if (q instanceof OpenSearchToParentBlockJoinQuery) {
                        nestedQuery.add(((OpenSearchToParentBlockJoinQuery) q).getChildQuery());
                    } else {
                        q.visit(this);
                    }
                }
            }
            return this;
        }
    }
}
