/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import lombok.AllArgsConstructor;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.search.SearchResponseSections;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.search.SearchHit;
import org.opensearch.search.SearchHits;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.PipelineProcessingContext;
import org.opensearch.search.pipeline.ProcessorGenerationContext;
import org.opensearch.search.pipeline.SearchResponseProcessor;
import org.opensearch.search.pipeline.SystemGeneratedProcessor;
import org.opensearch.search.profile.SearchProfileShardResults;

import java.io.IOException;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.MMR_RERANK_CONTEXT;
import static org.opensearch.knn.search.processor.mmr.MMRUtil.extractVectorFromHit;
import static org.opensearch.knn.search.processor.mmr.MMRUtil.shouldGenerateMMRProcessor;

/**
 * A system generated search response processor that rerank the response based on the Maximal Marginal Relevance
 */
@AllArgsConstructor
public class MMRRerankProcessor implements SearchResponseProcessor, SystemGeneratedProcessor {
    public static final String TYPE = "mmr_rerank";
    public static final String DESCRIPTION = "This is a system generated processor that will rerank the response based"
        + "on Maximal Marginal Relevance.";
    private final String tag;
    private final boolean ignoreFailure;

    @Override
    public SearchResponse processResponse(SearchRequest request, SearchResponse response) {
        throw new UnsupportedOperationException(
            String.format(Locale.ROOT, "Should not try to use %s to process a search response without PipelineProcessingContext.", TYPE)
        );
    }

    @Override
    public SearchResponse processResponse(SearchRequest request, SearchResponse searchResponse, PipelineProcessingContext requestContext)
        throws IOException {
        long startNanos = System.nanoTime();

        if (isEmptyResponse(searchResponse)) {
            return searchResponse;
        }

        final MMRRerankContext mmrContext = requireMMRContext(requestContext);
        final KNNVectorSimilarityFunction similarityFunction = mmrContext.getSpaceType().getKnnVectorSimilarityFunction();
        final int originalQuerySize = mmrContext.getOriginalQuerySize();
        final float diversity = mmrContext.getDiversity();
        final boolean isFloatVector = VectorDataType.FLOAT.equals(mmrContext.getVectorDataType());

        final List<SearchHit> candidates = new ArrayList<>(List.of(searchResponse.getHits().getHits()));
        final Map<Integer, Object> docVectors = extractVectors(
            candidates,
            mmrContext.getVectorFieldPath(),
            mmrContext.getIndexToVectorFieldPathMap(),
            isFloatVector
        );

        final List<SearchHit> selected = selectHitsWithMMR(
            candidates,
            docVectors,
            similarityFunction,
            diversity,
            originalQuerySize,
            isFloatVector
        );

        applyFetchSourceFilterIfNeeded(selected, mmrContext);

        final float maxSelectedScore = selected.stream().map(SearchHit::getScore).max(Float::compare).orElse(Float.NEGATIVE_INFINITY);

        final SearchHits newHits = new SearchHits(
            selected.toArray(new SearchHit[0]),
            searchResponse.getHits().getTotalHits(),
            maxSelectedScore,
            searchResponse.getHits().getSortFields(),
            searchResponse.getHits().getCollapseField(),
            searchResponse.getHits().getCollapseValues()
        );

        final SearchResponseSections newSections = new SearchResponseSections(
            newHits,
            searchResponse.getAggregations(),
            searchResponse.getSuggest(),
            searchResponse.isTimedOut(),
            searchResponse.isTerminatedEarly(),
            new SearchProfileShardResults(searchResponse.getProfileResults()),
            searchResponse.getNumReducePhases(),
            searchResponse.getInternalResponse().getSearchExtBuilders()
        );

        long elapsedMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
        long newTookMillis = searchResponse.getTook().millis() + elapsedMillis;

        return new SearchResponse(
            newSections,
            searchResponse.getScrollId(),
            searchResponse.getTotalShards(),
            searchResponse.getSuccessfulShards(),
            searchResponse.getSkippedShards(),
            newTookMillis,
            searchResponse.getPhaseTook(),
            searchResponse.getShardFailures(),
            searchResponse.getClusters(),
            searchResponse.pointInTimeId()
        );
    }

    private boolean isEmptyResponse(SearchResponse response) {
        return response == null
            || response.getHits() == null
            || response.getHits().getHits() == null
            || response.getHits().getHits().length == 0;
    }

    private MMRRerankContext requireMMRContext(PipelineProcessingContext requestContext) {
        Object attr = requestContext.getAttribute(MMR_RERANK_CONTEXT);
        if (attr == null) {
            throw new IllegalStateException("MMR rerank context cannot be null");
        }

        final MMRRerankContext ctx = (MMRRerankContext) attr;

        if (ctx.getSpaceType() == null) {
            throw new IllegalStateException("Space type in MMR rerank context cannot be null");
        }
        if (ctx.getOriginalQuerySize() == null) {
            throw new IllegalStateException("Original query size in MMR rerank context cannot be null");
        }
        if (ctx.getDiversity() == null) {
            throw new IllegalStateException("Diversity in MMR rerank context cannot be null");
        }
        if (ctx.getVectorDataType() == null) {
            throw new IllegalStateException("Vector data type in MMR rerank context cannot be null");
        }

        return ctx;
    }

    private Map<Integer, Object> extractVectors(
        List<SearchHit> hits,
        String defaultVectorFieldPath,
        Map<String, String> indexToVectorFieldPathMap,
        boolean isFloatVector
    ) {
        Map<Integer, Object> vectors = new ConcurrentHashMap<>();

        hits.parallelStream().forEach(hit -> {
            String vectorPath = defaultVectorFieldPath;

            if (indexToVectorFieldPathMap != null) {
                String overridePath = indexToVectorFieldPathMap.get(hit.getIndex());
                if (overridePath != null && !overridePath.isBlank()) {
                    vectorPath = overridePath;
                }
            }

            Object embedding = extractVectorFromHit(hit.getSourceAsMap(), vectorPath, hit.getId(), isFloatVector);
            vectors.put(hit.docId(), embedding);
        });

        return vectors;
    }

    private List<SearchHit> selectHitsWithMMR(
        List<SearchHit> candidates,
        Map<Integer, Object> docVectors,
        KNNVectorSimilarityFunction similarityFunction,
        float diversity,
        int targetSize,
        boolean isFloatVector
    ) {
        List<SearchHit> selected = new ArrayList<>();
        Map<Long, Float> simCache = new ConcurrentHashMap<>();

        while (selected.size() < targetSize && !candidates.isEmpty()) {

            Optional<SearchHit> bestCandidateOpt = candidates.parallelStream().max(Comparator.comparingDouble(candidate -> {
                int candidateId = candidate.docId();
                float maxSimToSelected = 0.0f;

                for (SearchHit sel : selected) {
                    int selId = sel.docId();
                    long key = cacheKey(candidateId, selId);
                    long symKey = cacheKey(selId, candidateId);

                    float sim = simCache.computeIfAbsent(key, k -> {
                        if (isFloatVector) {
                            return similarityFunction.compare((float[]) docVectors.get(candidateId), (float[]) docVectors.get(selId));
                        } else {
                            return similarityFunction.compare((byte[]) docVectors.get(candidateId), (byte[]) docVectors.get(selId));
                        }
                    });

                    simCache.putIfAbsent(symKey, sim);
                    maxSimToSelected = Math.max(maxSimToSelected, sim);
                }

                return (1 - diversity) * candidate.getScore() - diversity * maxSimToSelected;
            }));

            if (bestCandidateOpt.isPresent()) {
                SearchHit bestHit = bestCandidateOpt.get();
                selected.add(bestHit);
                candidates.remove(bestHit);
            }
        }

        return selected;
    }

    private void applyFetchSourceFilterIfNeeded(List<SearchHit> hits, MMRRerankContext mmrContext) throws IOException {
        final FetchSourceContext fetchSourceContext = mmrContext.getOriginalFetchSourceContext();
        if (fetchSourceContext == null) {
            return;
        }
        // if fetch source is false we directly remove the whole _source
        if (fetchSourceContext.fetchSource() == false) {
            for (SearchHit hit : hits) {
                hit.sourceRef(null);
            }
            return;
        }

        final Function<Map<String, ?>, Map<String, Object>> filter = fetchSourceContext.getFilter();
        for (SearchHit hit : hits) {
            Map<String, Object> filtered = filter.apply(hit.getSourceAsMap());
            hit.sourceRef(BytesReference.bytes(XContentFactory.jsonBuilder().map(filtered)));
        }
    }

    private long cacheKey(int id1, int id2) {
        return ((long) id1 << 32) | (id2 & 0xffffffffL);
    }

    // This processor will be executed pre the user defined search request processor if there is any. Since
    // we oversample before so it is better to execute this processor to rerank and reduce the response to the
    // original query size before executing other user defined search response processors.
    @Override
    public ExecutionStage getExecutionStage() {
        return ExecutionStage.PRE_USER_DEFINED;
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

    public static class MMRRerankProcessorFactory implements SystemGeneratedFactory<SearchResponseProcessor> {
        public static final String TYPE = "mmr_rerank_factory";

        @Override
        public boolean shouldGenerate(ProcessorGenerationContext context) {
            return shouldGenerateMMRProcessor(context);
        }

        @Override
        public SearchResponseProcessor create(
            Map<String, Factory<SearchResponseProcessor>> processorFactories,
            String tag,
            String description,
            boolean ignoreFailure,
            Map<String, Object> config,
            PipelineContext pipelineContext
        ) throws Exception {
            return new MMRRerankProcessor(tag, ignoreFailure);
        }
    }
}
