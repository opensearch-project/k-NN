/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor;

import com.google.common.annotations.VisibleForTesting;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.BooleanClause;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.index.query.InnerHitBuilder;
import org.opensearch.index.query.NestedQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilderVisitor;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.search.fetch.StoredFieldsContext;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.AbstractProcessor;
import org.opensearch.search.pipeline.Processor;
import org.opensearch.search.pipeline.ProcessorGenerationContext;
import org.opensearch.search.pipeline.SearchRequestProcessor;
import org.opensearch.search.pipeline.SystemGeneratedProcessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * A system-generated search request processor that automatically adds _source excludes for knn vector fields
 * found in the index mappings. This avoids returning large vector data in search responses by default.
 * User-provided includes are always preserved.
 */
@Log4j2
public final class KNNSourceExcludesProcessor extends AbstractProcessor implements SearchRequestProcessor, SystemGeneratedProcessor {

    public static final String TYPE = "knn_default_excludes";

    private final ClusterService clusterService;
    private final IndexNameExpressionResolver indexNameExpressionResolver;

    KNNSourceExcludesProcessor(
        String tag,
        String description,
        boolean ignoreFailure,
        ClusterService clusterService,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        super(tag, description, ignoreFailure);
        this.clusterService = clusterService;
        this.indexNameExpressionResolver = indexNameExpressionResolver;
    }

    @Override
    public SearchRequest processRequest(SearchRequest request) {
        final Set<String> vectorFields = getVectorFieldsFromMappings(request);
        if (vectorFields.isEmpty()) {
            return request;
        }

        final InnerHitsExtractor innerHitsExtractor = new InnerHitsExtractor();
        if (request.source().query() != null) {
            request.source().query().visit(innerHitsExtractor);
        }

        final List<InnerHitBuilder> innerHitBuilders = innerHitsExtractor.getInnerHitBuilders();
        for (InnerHitBuilder innerHitBuilder : innerHitBuilders) {
            FetchSourceContext fetchSourceContext = innerHitBuilder.getFetchSourceContext();
            List<String> fetchFields = innerHitBuilder.getFetchFields() != null
                ? innerHitBuilder.getFetchFields().stream().map(fieldAndFormat -> fieldAndFormat.field).toList()
                : List.of();

            if (sourceExplicitTrue(fetchSourceContext)
                && Stream.concat(Arrays.stream(fetchSourceContext.includes()), fetchFields.stream()).anyMatch(vectorFields::contains)) {
                // If source is explicitly true for inner hits we should not apply excludes at top level as it will return
                // mask 1 in inner hits. https://github.com/opensearch-project/k-NN/issues/3303
                return request;
            }

            innerHitBuilder.setFetchSourceContext(applyExcludes(innerHitBuilder.getFetchSourceContext(), vectorFields));
        }

        request.source().fetchSource(applyExcludes(request.source().fetchSource(), vectorFields));
        return request;
    }

    /**
     * Collect all knn_vector field names from the index mappings for the target indices.
     */
    private Set<String> getVectorFieldsFromMappings(SearchRequest request) {
        String[] indices = request.indices();
        if (indices == null || indices.length == 0) {
            return Set.of();
        }

        ClusterState state = clusterService.state();
        String[] concreteIndices = indexNameExpressionResolver.concreteIndexNames(state, request);
        Set<String> vectorFields = new HashSet<>();

        for (String indexName : concreteIndices) {
            IndexMetadata indexMetadata = state.metadata().index(indexName);
            if (indexMetadata != null) {
                MappingMetadata mappingMetadata = indexMetadata.mapping();
                if (mappingMetadata != null) {
                    collectVectorFields(mappingMetadata.sourceAsMap(), "", vectorFields);
                }
            }
        }
        return vectorFields;
    }

    /**
     * Recursively walk the mapping tree and collect field names with type "knn_vector".
     */
    @SuppressWarnings("unchecked")
    @VisibleForTesting
    static void collectVectorFields(final Map<String, Object> mappingMap, final String prefix, final Set<String> vectorFields) {
        final Map<String, Object> properties = (Map<String, Object>) mappingMap.get("properties");
        if (properties == null) {
            return;
        }
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            if (entry.getValue() instanceof Map) {
                final Map<String, Object> fieldMapping = (Map<String, Object>) entry.getValue();
                final String fullName = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
                if (KNNVectorFieldMapper.CONTENT_TYPE.equals(fieldMapping.get("type"))) {
                    vectorFields.add(fullName);
                }
                // Recurse into nested objects
                if (fieldMapping.containsKey("properties")) {
                    collectVectorFields(fieldMapping, fullName, vectorFields);
                }
            }
        }
    }

    private FetchSourceContext applyExcludes(FetchSourceContext current, Set<String> vectorFields) {
        String[] userIncludes = (current != null) ? current.includes() : new String[0];
        String[] userExcludes = (current != null) ? current.excludes() : new String[0];

        Set<String> includeSet = (userIncludes.length > 0) ? Set.of(userIncludes) : Set.of();
        Set<String> mergedExcludes = (userExcludes.length > 0) ? new HashSet<>(Arrays.asList(userExcludes)) : new HashSet<>();

        for (String vf : vectorFields) {
            if (!includeSet.contains(vf)) {
                mergedExcludes.add(vf);
            }
        }
        return new FetchSourceContext(true, userIncludes, mergedExcludes.toArray(new String[0]));
    }

    @Override
    public String getType() {
        return TYPE;
    }

    private static boolean sourceExplicitTrue(FetchSourceContext fetchSource) {
        return fetchSource != null
            && fetchSource.fetchSource()
            && (fetchSource.includes().length == 0 && fetchSource.excludes().length == 0);
    }

    public static class Factory implements SystemGeneratedProcessor.SystemGeneratedFactory<SearchRequestProcessor> {
        public static final String TYPE = "knn_default_excludes_factory";

        private final ClusterService clusterService;
        private final IndexNameExpressionResolver indexNameExpressionResolver;

        public Factory(ClusterService clusterService, IndexNameExpressionResolver indexNameExpressionResolver) {
            this.clusterService = clusterService;
            this.indexNameExpressionResolver = indexNameExpressionResolver;
        }

        @Override
        public boolean shouldGenerate(ProcessorGenerationContext context) {
            // TODO: Access parent action and use an allowlisted list of action to generate the processor
            // For now, we will add this processor for all search requests except parent task requests
            if (context.searchRequest() == null || context.searchRequest().getParentTask().isSet()) {
                return false;
            }

            final SearchRequest request = context.searchRequest();
            if (request.source() != null) {
                final FetchSourceContext fetchSource = request.source().fetchSource();
                final StoredFieldsContext storedFieldsContext = request.source().storedFields();

                if (shouldAppendExcludes(fetchSource, storedFieldsContext) == false) {
                    return false;
                }

                return true;
            }
            // In all other cases do not add this processor
            return false;
        }

        private boolean sourceExplicitFalse(FetchSourceContext fetchSource) {
            return fetchSource != null && fetchSource.fetchSource() == false;
        }

        private boolean shouldAppendExcludes(FetchSourceContext fetchSource, StoredFieldsContext storedFieldsContext) {
            if (sourceExplicitFalse(fetchSource) || sourceExplicitTrue(fetchSource)) {
                return false;
            }

            if (storedFieldsContext != null && storedFieldsContext.fetchFields() == false) {
                return false;
            }

            return true;
        }

        @Override
        public SearchRequestProcessor create(
            Map<String, Processor.Factory<SearchRequestProcessor>> processorFactories,
            String tag,
            String description,
            boolean ignoreFailure,
            Map<String, Object> config,
            PipelineContext pipelineContext
        ) {
            return new KNNSourceExcludesProcessor(tag, description, ignoreFailure, clusterService, indexNameExpressionResolver);
        }
    }

    @Getter
    private static class InnerHitsExtractor implements QueryBuilderVisitor {

        private final List<InnerHitBuilder> innerHitBuilders = new ArrayList<>();

        @Override
        public void accept(QueryBuilder qb) {
            if (qb instanceof NestedQueryBuilder nestedQueryBuilder && nestedQueryBuilder.innerHit() != null) {
                innerHitBuilders.add(nestedQueryBuilder.innerHit());
            }
        }

        @Override
        public QueryBuilderVisitor getChildVisitor(BooleanClause.Occur occur) {
            return this;
        }
    }
}
