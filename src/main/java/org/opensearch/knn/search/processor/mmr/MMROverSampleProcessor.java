/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import lombok.AllArgsConstructor;
import org.opensearch.action.OriginalIndices;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.opensearch.knn.search.extension.MMRSearchExtBuilder;
import org.opensearch.search.fetch.StoredFieldsContext;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.PipelineProcessingContext;
import org.opensearch.search.pipeline.ProcessorGenerationContext;
import org.opensearch.search.pipeline.SearchRequestProcessor;
import org.opensearch.search.pipeline.SystemGeneratedProcessor;
import org.opensearch.transport.RemoteClusterService;
import org.opensearch.transport.client.Client;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.MMR_RERANK_CONTEXT;
import static org.opensearch.knn.search.processor.mmr.MMRUtil.resolveKnnVectorFieldInfo;
import static org.opensearch.knn.search.processor.mmr.MMRUtil.shouldGenerateMMRProcessor;

/**
 * A system generated search request processor for MMR. It will transform the search request to oversample and also
 * collect and store the info in the PipelineProcessingContext for MMR later in a search response processor.
 */
public class MMROverSampleProcessor implements SearchRequestProcessor, SystemGeneratedProcessor {
    public static final String TYPE = "mmr_over_sample";
    public static final String DESCRIPTION = "This is a system generated processor that will modify the query size and"
        + "k of the knn query to oversample for Maximal Marginal Relevance rerank.";
    private static final int DEFAULT_QUERY_SIZE_INDICATOR = -1;
    private static final int DEFAULT_QUERY_SIZE = 10;
    private static final int DEFAULT_OVERSAMPLE_SCALE = 3;
    private final String tag;
    private final boolean ignoreFailure;
    private final Client client;
    private final Map<String, MMRQueryTransformer<? extends QueryBuilder>> mmrQueryTransformers;

    public MMROverSampleProcessor(
        String tag,
        boolean ignoreFailure,
        Client client,
        Map<String, MMRQueryTransformer<? extends QueryBuilder>> mmrQueryTransformers
    ) {
        this.tag = tag;
        this.ignoreFailure = ignoreFailure;
        this.client = client;
        this.mmrQueryTransformers = mmrQueryTransformers;
    }

    @Override
    public SearchRequest processRequest(SearchRequest searchRequest) {
        throw new UnsupportedOperationException(
            String.format(Locale.ROOT, "Should not try to use %s to process a search request synchronously.", TYPE)
        );
    }

    @Override
    public SearchRequest processRequest(SearchRequest request, PipelineProcessingContext requestContext) {
        throw new UnsupportedOperationException(
            String.format(Locale.ROOT, "Should not try to use %s to process a search request synchronously.", TYPE)
        );
    }

    @Override
    public void processRequestAsync(
        SearchRequest request,
        PipelineProcessingContext requestContext,
        ActionListener<SearchRequest> requestListener
    ) {
        try {
            if (request == null || request.source() == null || request.source().ext() == null) {
                throw new IllegalStateException(
                    String.format(Locale.ROOT, "Search request passed to %s search request processor must have mmr search extension.", TYPE)
                );
            }

            // Find the MMRSearchExtBuilder. We must have one.
            MMRSearchExtBuilder mmrSearchExtBuilder = extractMMRExtension(request);

            String[] allTargetIndices = request.indices();
            String remoteSeparator = String.valueOf(RemoteClusterService.REMOTE_CLUSTER_INDEX_SEPARATOR);
            List<String> remoteIndices = splitIndices(allTargetIndices, remoteSeparator, true);
            List<String> localIndices = splitIndices(allTargetIndices, remoteSeparator, false);

            MMRRerankContext mmrRerankContext = new MMRRerankContext();
            mmrRerankContext.setDiversity(mmrSearchExtBuilder.getDiversity());

            validateForRemoteIndices(mmrSearchExtBuilder, remoteIndices);

            int candidates = computeCandidatesAndSetRequestSize(mmrRerankContext, request, mmrSearchExtBuilder);
            // ensure we have the vector in the _source so that the MMRRerankProcessor can use it for mmr rerank
            preserveAndEnableFullSource(request, mmrRerankContext);

            OriginalIndices localIndicesSearchRequest = new OriginalIndices(localIndices.toArray(String[]::new), request.indicesOptions());
            List<IndexMetadata> localIndexMetadataList = getLocalIndexMetadata(localIndicesSearchRequest);
            String userProvidedVectorFieldPath = mmrSearchExtBuilder.getVectorFieldPath();
            VectorDataType userProvidedVectorDataType = mmrSearchExtBuilder.getVectorFieldDataType();
            SpaceType userProvidedSpaceType = mmrSearchExtBuilder.getSpaceType();
            MMRTransformContext mmrTransformContext = new MMRTransformContext(
                candidates,
                mmrRerankContext,
                localIndexMetadataList,
                remoteIndices,
                userProvidedSpaceType,
                userProvidedVectorFieldPath,
                userProvidedVectorDataType,
                client,
                false
            );

            if (userProvidedVectorFieldPath != null) {
                processWithUserProvidedVectorFieldPath(request, requestContext, requestListener, mmrTransformContext);
                return;
            }
            transformQueryForMMR(request, requestListener, mmrTransformContext, requestContext);
        } catch (Exception e) {
            requestListener.onFailure(e);
        }
    }

    private void processWithUserProvidedVectorFieldPath(
        SearchRequest request,
        PipelineProcessingContext requestContext,
        ActionListener<SearchRequest> requestListener,
        MMRTransformContext mmrTransformContext
    ) {
        try {
            String userProvidedVectorFieldPath = mmrTransformContext.getUserProvidedVectorFieldPath();
            SpaceType userProvidedSpaceType = mmrTransformContext.getUserProvidedSpaceType();
            VectorDataType userProvidedVectorDataType = mmrTransformContext.getUserProvidedVectorDataType();
            List<IndexMetadata> localIndexMetadataList = mmrTransformContext.getLocalIndexMetadataList();
            MMRRerankContext mmrRerankContext = mmrTransformContext.getMmrRerankContext();

            mmrRerankContext.setVectorFieldPath(userProvidedVectorFieldPath);

            resolveKnnVectorFieldInfo(
                userProvidedVectorFieldPath,
                userProvidedSpaceType,
                userProvidedVectorDataType,
                localIndexMetadataList,
                client,
                ActionListener.wrap(vectorFieldInfo -> {
                    mmrRerankContext.setVectorDataType(vectorFieldInfo.getVectorDataType());
                    mmrRerankContext.setSpaceType(vectorFieldInfo.getSpaceType());
                    mmrTransformContext.setVectorFieldInfoResolved(true);
                    transformQueryForMMR(request, requestListener, mmrTransformContext, requestContext);
                }, requestListener::onFailure)
            );
        } catch (Exception e) {
            requestListener.onFailure(e);
        }
    }

    private MMRSearchExtBuilder extractMMRExtension(SearchRequest request) {
        return request.source()
            .ext()
            .stream()
            .filter(MMRSearchExtBuilder.class::isInstance)
            .map(MMRSearchExtBuilder.class::cast)
            .findFirst()
            .orElseThrow(
                () -> new IllegalStateException(
                    String.format(Locale.ROOT, "SearchRequest passed to %s processor must have an MMRSearchExtBuilder", TYPE)
                )
            );
    }

    private List<String> splitIndices(String[] indices, String separator, boolean remote) {
        return Arrays.stream(indices).filter(index -> (index.contains(separator)) == remote).toList();
    }

    // For remote indices it is not cheap to pull the info from the remote cluster to resolve the space type and the
    // vector data type so we require users to provide this info.
    private void validateForRemoteIndices(MMRSearchExtBuilder mmrSearchExtBuilder, List<String> remoteIndices) {
        if (remoteIndices.isEmpty()) {
            return;
        }

        String indicesString = String.join(",", remoteIndices);

        SpaceType spaceType = mmrSearchExtBuilder.getSpaceType();
        if (spaceType == null) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "%s is required in the MMR query extension when querying remote indices [%s].",
                    MMRSearchExtBuilder.VECTOR_FIELD_SPACE_TYPE_FIELD.getPreferredName(),
                    indicesString
                )
            );
        }

        VectorDataType vectorDataType = mmrSearchExtBuilder.getVectorFieldDataType();
        if (vectorDataType == null) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "%s is required in the MMR query extension when querying remote indices [%s].",
                    MMRSearchExtBuilder.VECTOR_FIELD_DATA_TYPE_FIELD.getPreferredName(),
                    indicesString
                )
            );
        }
    }

    private List<IndexMetadata> getLocalIndexMetadata(OriginalIndices localIndicesSearchRequest) {
        return KNNClusterUtil.instance().getIndexMetadataList(localIndicesSearchRequest);
    }

    private int computeCandidatesAndSetRequestSize(
        MMRRerankContext mmrRerankContext,
        SearchRequest request,
        MMRSearchExtBuilder mmrSearchExtBuilder
    ) {
        int originalQuerySize = request.source().size();
        // If the query size is not set by the user then it will be -1, and later we will use the default value 10
        if (originalQuerySize == DEFAULT_QUERY_SIZE_INDICATOR) {
            originalQuerySize = DEFAULT_QUERY_SIZE;
        }
        mmrRerankContext.setOriginalQuerySize(originalQuerySize);

        Integer candidates = mmrSearchExtBuilder.getCandidates();
        if (candidates == null) {
            candidates = DEFAULT_OVERSAMPLE_SCALE * originalQuerySize; // default candidates
        }

        request.source().size(candidates);
        return candidates;
    }

    private void preserveAndEnableFullSource(SearchRequest request, MMRRerankContext mmrContext) {
        FetchSourceContext currentSourceContext = request.source().fetchSource();
        StoredFieldsContext storedFieldsContext = request.source().storedFields();

        if (storedFieldsContext != null) {
            if (isStoredFieldsDisabled(storedFieldsContext)) {
                handleDisabledStoredFields(request, mmrContext, currentSourceContext);
                return;
            }

            if (isSourceNotExplicitlySet(currentSourceContext)) {
                // when stored_fields is defined and _source is not defined we will not fetch _source so need to
                // temporarily enable it for mmr.
                enableSourceTemporarily(request, mmrContext);
                return;
            }
        }

        if (isAlreadyFetchingFullSource(currentSourceContext)) {
            return;
        }

        preserveAndEnableFullSourceFetch(request, mmrContext, currentSourceContext);
    }

    private boolean isStoredFieldsDisabled(StoredFieldsContext context) {
        return context.fetchFields() == false;
    }

    private boolean isSourceNotExplicitlySet(FetchSourceContext sourceContext) {
        return sourceContext == null;
    }

    private boolean isAlreadyFetchingFullSource(FetchSourceContext sourceContext) {
        if (sourceContext == null) {
            return true;
        }
        boolean fetchingAll = sourceContext.fetchSource();
        boolean noIncludes = sourceContext.includes().length == 0;
        boolean noExcludes = sourceContext.excludes().length == 0;
        return fetchingAll && noIncludes && noExcludes;
    }

    private void handleDisabledStoredFields(SearchRequest request, MMRRerankContext mmrContext, FetchSourceContext currentSourceContext) {
        if (currentSourceContext != null) {
            // stored_fields = _none_ + explicit _source → invalid
            throw new IllegalArgumentException("[stored_fields] cannot be disabled if [_source] is requested");
        }
        // stored_fields = _none_ + no _source defined → temporarily enable _source
        mmrContext.setOriginalFetchSourceContext(new FetchSourceContext(false));
        request.source().storedFields(StoredFieldsContext.fromList(Collections.emptyList()));
        request.source().fetchSource(new FetchSourceContext(true));
    }

    private void enableSourceTemporarily(SearchRequest request, MMRRerankContext mmrContext) {
        mmrContext.setOriginalFetchSourceContext(new FetchSourceContext(false));
        request.source().fetchSource(new FetchSourceContext(true));
    }

    private void preserveAndEnableFullSourceFetch(
        SearchRequest request,
        MMRRerankContext mmrContext,
        FetchSourceContext currentSourceContext
    ) {
        mmrContext.setOriginalFetchSourceContext(currentSourceContext);
        request.source().fetchSource(new FetchSourceContext(true));
    }

    private void transformQueryForMMR(
        SearchRequest request,
        ActionListener<SearchRequest> requestListener,
        MMRTransformContext mmrTransformationContext,
        PipelineProcessingContext requestContext
    ) {
        QueryBuilder queryBuilder = request.source().query();
        if (queryBuilder == null) {
            throw new IllegalArgumentException("Query builder must not be null to do Maximal Marginal Relevance rerank.");
        }

        @SuppressWarnings("unchecked")
        MMRQueryTransformer<QueryBuilder> transformer = (MMRQueryTransformer<QueryBuilder>) mmrQueryTransformers.get(
            queryBuilder.getWriteableName()
        );
        if (transformer == null) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Maximal Marginal Relevance rerank doesn't support the query type [%s]",
                    queryBuilder.getWriteableName()
                )
            );
        }

        transformer.transform(queryBuilder, new ActionListener<>() {
            @Override
            public void onResponse(Void unused) {
                requestContext.setAttribute(MMR_RERANK_CONTEXT, mmrTransformationContext.getMmrRerankContext());
                requestListener.onResponse(request);
            }

            @Override
            public void onFailure(Exception e) {
                requestListener.onFailure(e);
            }
        }, mmrTransformationContext);
    }

    // This processor will be executed post the user defined search request processor if there is any.
    @Override
    public ExecutionStage getExecutionStage() {
        return ExecutionStage.POST_USER_DEFINED;
    }

    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public String getTag() {
        return tag;
    }

    @Override
    public String getDescription() {
        return DESCRIPTION;
    }

    @Override
    public boolean isIgnoreFailure() {
        return ignoreFailure;
    }

    @AllArgsConstructor
    public static class MMROverSampleProcessorFactory implements SystemGeneratedFactory<SearchRequestProcessor> {
        public static final String TYPE = "mmr_over_sample_factory";
        private final Client client;
        private final Map<String, MMRQueryTransformer<? extends QueryBuilder>> mmrQueryTransformers;

        @Override
        public boolean shouldGenerate(ProcessorGenerationContext processorGenerationContext) {
            return shouldGenerateMMRProcessor(processorGenerationContext);
        }

        @Override
        public SearchRequestProcessor create(
            Map<String, Factory<SearchRequestProcessor>> processorFactories,
            String tag,
            String description,
            boolean ignoreFailure,
            Map<String, Object> config,
            PipelineContext pipelineContext
        ) throws Exception {
            return new MMROverSampleProcessor(tag, ignoreFailure, client, mmrQueryTransformers);
        }
    }
}
