/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor;

import com.google.common.annotations.VisibleForTesting;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.AbstractProcessor;
import org.opensearch.search.pipeline.Processor;
import org.opensearch.search.pipeline.SearchRequestProcessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A search request processor that automatically adds _source excludes for knn vector fields
 * found in the query. This avoids returning large vector data in search responses by default.
 * User-provided includes are always preserved.
 */
public final class KNNDefaultExcludesProcessor extends AbstractProcessor implements SearchRequestProcessor {

    public static final String TYPE = "knn_default_excludes";

    KNNDefaultExcludesProcessor(String tag, String description, boolean ignoreFailure) {
        super(tag, description, ignoreFailure);
    }

    @Override
    public SearchRequest processRequest(SearchRequest request) {
        if (request.source() == null || request.source().query() == null) {
            return request;
        }

        final Set<String> vectorFields = new HashSet<>();
        collectKnnFieldNames(request.source().query(), vectorFields);

        if (vectorFields.isEmpty()) {
            return request;
        }

        FetchSourceContext current = request.source().fetchSource();
        String[] userIncludes = (current != null) ? current.includes() : new String[0];
        String[] userExcludes = (current != null) ? current.excludes() : new String[0];
        boolean fetchSource = (current == null) || current.fetchSource();

        Set<String> includeSet = new HashSet<>(Arrays.asList(userIncludes));
        Set<String> mergedExcludes = new HashSet<>(Arrays.asList(userExcludes));
        for (String vf : vectorFields) {
            if (!includeSet.contains(vf)) {
                mergedExcludes.add(vf);
            }
        }

        request.source().fetchSource(new FetchSourceContext(fetchSource, userIncludes, mergedExcludes.toArray(new String[0])));
        return request;
    }

    @Override
    public String getType() {
        return TYPE;
    }

    /**
     * Recursively walk the query tree and collect field names from KNNQueryBuilder instances.
     */
    @VisibleForTesting
    static void collectKnnFieldNames(QueryBuilder query, Set<String> fieldNames) {
        if (query instanceof KNNQueryBuilder) {
            fieldNames.add(((KNNQueryBuilder) query).fieldName());
        } else if (query instanceof BoolQueryBuilder) {
            BoolQueryBuilder bool = (BoolQueryBuilder) query;
            List<QueryBuilder> clauses = new ArrayList<>();
            clauses.addAll(bool.must());
            clauses.addAll(bool.should());
            clauses.addAll(bool.filter());
            clauses.addAll(bool.mustNot());
            for (QueryBuilder clause : clauses) {
                collectKnnFieldNames(clause, fieldNames);
            }
        }
    }

    public static class Factory implements Processor.Factory<SearchRequestProcessor> {
        @Override
        public SearchRequestProcessor create(
            Map<String, Processor.Factory<SearchRequestProcessor>> processorFactories,
            String tag,
            String description,
            boolean ignoreFailure,
            Map<String, Object> config,
            PipelineContext pipelineContext
        ) {
            return new KNNDefaultExcludesProcessor(tag, description, ignoreFailure);
        }
    }
}
