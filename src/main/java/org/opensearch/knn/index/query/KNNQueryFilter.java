/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.opensearch.index.query.QueryBuilder;

import java.util.ArrayList;
import java.util.List;

@NoArgsConstructor
@Getter
public class KNNQueryFilter {

    List<QueryBuilder> mustClauses = new ArrayList<>();
    List<QueryBuilder> shouldClauses = new ArrayList<>();
    List<QueryBuilder> mustNotClauses = new ArrayList<>();

    public KNNQueryFilter must(QueryBuilder queryBuilder) {
        if (queryBuilder == null) {
            throw new IllegalArgumentException("inner bool query clause cannot be null");
        }
        mustClauses.add(queryBuilder);
        return this;
    }

    public KNNQueryFilter should(QueryBuilder queryBuilder) {
        if (queryBuilder == null) {
            throw new IllegalArgumentException("inner bool query clause cannot be null");
        }
        shouldClauses.add(queryBuilder);
        return this;
    }

    public KNNQueryFilter mustNot(QueryBuilder queryBuilder) {
        if (queryBuilder == null) {
            throw new IllegalArgumentException("inner bool query clause cannot be null");
        }
        mustNotClauses.add(queryBuilder);
        return this;
    }
}
