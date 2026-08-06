/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor;

import com.google.common.annotations.VisibleForTesting;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.BooleanClause;
import org.opensearch.action.search.MultiSearchAction;
import org.opensearch.action.search.SearchAction;
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
import org.opensearch.common.regex.Regex;
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

/**
 * A system-generated search request processor that automatically excludes {@code knn_vector} fields
 * from {@code _source} in search responses. This prevents large vector arrays from being returned
 * by default, reducing response payload size without requiring users to manually configure
 * {@code _source} filtering.
 *
 * <p>The processor is injected into the search pipeline before user-defined processors
 * ({@link ExecutionStage#PRE_USER_DEFINED}). It reads the index mappings at request time to discover
 * all {@code knn_vector} fields, then appends them as {@code _source.excludes} on the request.
 *
 * <p><b>Skipped entirely when:</b>
 * <ul>
 *   <li>The request has no target indices, or none resolve to a concrete index.</li>
 *   <li>The index mapping has {@code "_source": {"enabled": false}} — nothing to filter.</li>
 *   <li>The request explicitly disables source ({@code _source: false}) or fully enables it
 *       ({@code _source: true} with no includes/excludes).</li>
 *   <li>{@code stored_fields: _none_} — source is suppressed at the transport layer.</li>
 *   <li>Any inner hit has {@code _source: true} (fetch everything) or explicitly requests
 *       a vector field via {@code _source.includes} or {@code fields} — adding top-level excludes
 *       would mask the field in inner hit responses (see
 *       <a href="https://github.com/opensearch-project/k-NN/issues/3303">issue 3303</a>).</li>
 * </ul>
 *
 * <p><b>Excludes are not added for a vector field when:</b>
 * <ul>
 *   <li>The user's {@code _source.includes} already covers the field (exact or glob pattern).</li>
 *   <li>The user's {@code _source.excludes} already covers the field (exact or glob pattern).</li>
 *   <li>The index mapping's {@code _source.excludes} already covers the field.</li>
 * </ul>
 *
 * <p>Inner hits whose source is explicitly disabled ({@code _source: false}) are left unchanged;
 * excludes are only applied to inner hits whose source is unconstrained or has existing filters.
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
    public ExecutionStage getExecutionStage() {
        return ExecutionStage.PRE_USER_DEFINED;
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
            StoredFieldsContext storedFieldsContext = innerHitBuilder.getStoredFieldsContext();

            if (shouldAppendExcludes(fetchSourceContext, storedFieldsContext)) {
                innerHitBuilder.setFetchSourceContext(applyExcludes(innerHitBuilder.getFetchSourceContext(), vectorFields));
            }
        }

        request.source().fetchSource(applyExcludes(request.source().fetchSource(), vectorFields));
        return request;
    }

    /**
     * Resolves concrete index names from the request (handling aliases, wildcards, date-math)
     * and collects all exposed {@code knn_vector} field paths across those indices.
     * Fields whose index has {@code _source} disabled, or that are already covered by the
     * index-level {@code _source.excludes}, are not included.
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
                collectExposedVectorFields(indexMetadata, vectorFields);
            }
        }
        return vectorFields;
    }

    /**
     * Adds all {@code knn_vector} field paths from the given index to {@code vectorFields},
     * skipping the index if {@code _source} is disabled, and filtering out fields already
     * covered by the index-level {@code _source.excludes}.
     */
    private static void collectExposedVectorFields(final IndexMetadata indexMetadata, final Set<String> vectorFields) {
        final MappingMetadata mappingMetadata = indexMetadata.mapping();
        if (mappingMetadata == null) {
            return;
        }
        final Map<String, Object> mappingSource = mappingMetadata.sourceAsMap();
        if (SourceInspector.isEnabled(mappingSource) == false) {
            return;
        }
        collectVectorFields(mappingSource, "", vectorFields);
        vectorFields.removeIf(field -> SourceInspector.isAlreadyExcluded(field, mappingSource));
    }

    /**
     * Recursively walks the mapping {@code properties} tree and adds any field whose
     * {@code type} is {@code knn_vector} to {@code vectorFields}.
     * Nested and object fields are traversed by recursing into their {@code properties} block.
     *
     * @param mappingMap   the mapping node to inspect (top-level or a nested object block)
     * @param prefix       dot-separated path prefix for this node (empty string at the root)
     * @param vectorFields accumulator for discovered field paths
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
                if (fieldMapping.containsKey("properties")) {
                    collectVectorFields(fieldMapping, fullName, vectorFields);
                }
            }
        }
    }

    private FetchSourceContext applyExcludes(FetchSourceContext current, Set<String> vectorFields) {
        String[] userIncludes = (current != null) ? current.includes() : new String[0];
        String[] userExcludes = (current != null) ? current.excludes() : new String[0];

        Set<String> mergedExcludes = (userExcludes.length > 0) ? new HashSet<>(Arrays.asList(userExcludes)) : new HashSet<>();

        for (String vectorField : vectorFields) {
            if (SourceInspector.patternMatchesAny(vectorField, Arrays.asList(userIncludes)) == false
                && SourceInspector.patternMatchesAny(vectorField, Arrays.asList(userExcludes)) == false) {
                mergedExcludes.add(vectorField);
            }
        }
        return new FetchSourceContext(true, userIncludes, mergedExcludes.toArray(new String[0]));
    }

    private static boolean shouldAppendExcludes(FetchSourceContext fetchSource, StoredFieldsContext storedFieldsContext) {
        if (SourceInspector.isExplicitFalse(fetchSource) || SourceInspector.isExplicitTrue(fetchSource)) {
            return false;
        }
        return storedFieldsContext == null || storedFieldsContext.fetchFields();
    }

    @Override
    public String getType() {
        return TYPE;
    }

    /** Helpers for inspecting _source configuration at both the mapping and request level. */
    private static final class SourceInspector {

        private SourceInspector() {}

        @SuppressWarnings("unchecked")
        private static boolean isEnabled(final Map<String, Object> mappingMap) {
            final Object sourceMeta = mappingMap.get("_source");
            if (sourceMeta instanceof Map) {
                final Object enabled = ((Map<String, Object>) sourceMeta).get("enabled");
                if (Boolean.FALSE.equals(enabled)) {
                    return false;
                }
            }
            return true;
        }

        @SuppressWarnings("unchecked")
        private static boolean isAlreadyExcluded(final String fieldName, final Map<String, Object> mappingMap) {
            final Object sourceMeta = mappingMap.get("_source");
            if (!(sourceMeta instanceof Map)) {
                return false;
            }
            final Object excludes = ((Map<String, Object>) sourceMeta).get("excludes");
            if (!(excludes instanceof List)) {
                return false;
            }
            return patternMatchesAny(fieldName, (List<?>) excludes);
        }

        private static boolean isExplicitTrue(final FetchSourceContext fetchSource) {
            return fetchSource != null
                && fetchSource.fetchSource()
                && fetchSource.includes().length == 0
                && fetchSource.excludes().length == 0;
        }

        private static boolean isExplicitFalse(final FetchSourceContext fetchSource) {
            return fetchSource != null && fetchSource.fetchSource() == false;
        }

        private static boolean patternMatchesAny(final String fieldName, final List<?> patterns) {
            for (Object pattern : patterns) {
                if (pattern instanceof String p && Regex.simpleMatch(p, fieldName)) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * System-generated factory for {@link KNNSourceExcludesProcessor}.
     *
     * <p>{@link #shouldGenerate} is evaluated once per search request before the pipeline is built.
     * It returns {@code false} for requests where applying source excludes would be a no-op or
     * incorrect (parent-task sub-requests, source already disabled/fully-enabled, stored fields
     * suppressed). Inner-hit source constraints are evaluated at {@link #processRequest} time
     * since the factory does not have access to mapping data.
     */
    public static class Factory implements SystemGeneratedProcessor.SystemGeneratedFactory<SearchRequestProcessor> {
        public static final String TYPE = "knn_default_excludes_factory";

        /**
         * Parent actions for which this processor should be generated.
         * <p>A {@code null} parent action denotes a top-level user search with no parent task and is always allowed.
         * {@link SearchAction#NAME} covers the leaf search executed on a remote cluster during cross-cluster search;
         * {@link MultiSearchAction#NAME} covers the child searches fanned out from a multi-search request.
         */
        private static final Set<String> ALLOWED_PARENT_ACTIONS = Set.of(SearchAction.NAME, MultiSearchAction.NAME);

        private final ClusterService clusterService;
        private final IndexNameExpressionResolver indexNameExpressionResolver;

        public Factory(ClusterService clusterService, IndexNameExpressionResolver indexNameExpressionResolver) {
            this.clusterService = clusterService;
            this.indexNameExpressionResolver = indexNameExpressionResolver;
        }

        @Override
        public boolean shouldGenerate(ProcessorGenerationContext context) {
            if (context.searchRequest() == null) {
                return false;
            }

            // Only generate the processor for allowlisted parent actions. A null parent action is a top-level
            // user search; msearch and cross-cluster leaf searches are the other cases where excluding vector
            // fields from the response is meaningful. Any other parent action is skipped.
            if (context.parentAction() != null && ALLOWED_PARENT_ACTIONS.contains(context.parentAction()) == false) {
                return false;
            }

            final SearchRequest request = context.searchRequest();
            if (request.source() != null) {
                final FetchSourceContext fetchSource = request.source().fetchSource();
                final StoredFieldsContext storedFieldsContext = request.source().storedFields();

                return shouldAppendExcludes(fetchSource, storedFieldsContext);
            }
            // In all other cases do not add this processor
            return false;
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

    /**
     * Visitor that traverses the query tree and collects {@link InnerHitBuilder} instances
     * from all {@link NestedQueryBuilder} nodes that have inner hits configured.
     */
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
