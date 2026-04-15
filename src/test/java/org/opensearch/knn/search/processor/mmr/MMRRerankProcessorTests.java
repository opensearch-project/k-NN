/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.search.TotalHits;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.search.SearchResponseSections;
import org.opensearch.action.search.ShardSearchFailure;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.search.SearchHit;
import org.opensearch.search.SearchHits;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.PipelineProcessingContext;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.MMR_EXPLAIN;
import static org.opensearch.knn.common.KNNConstants.MMR_RERANK_CONTEXT;
import static org.mockito.Mockito.mock;

public class MMRRerankProcessorTests extends KNNTestCase {
    private MMRRerankProcessor processor;
    private SearchRequest searchRequest;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        processor = new MMRRerankProcessor("test-tag", false);
        searchRequest = new SearchRequest();
    }

    public void testProcessResponse_withoutContext_thenThrowsUnsupportedOperationException() {
        UnsupportedOperationException ex = assertThrows(
            UnsupportedOperationException.class,
            () -> processor.processResponse(searchRequest, mock(SearchResponse.class))
        );

        assertEquals("Should not try to use mmr_rerank to process a search response without PipelineProcessingContext.", ex.getMessage());
    }

    public void testProcessResponse_whenEmptyHits_thenReturnOriginalResponse() throws IOException {
        SearchResponse emptyResponse = createSearchResponse(new SearchHit[] {});

        PipelineProcessingContext ctx = mock(PipelineProcessingContext.class);

        SearchResponse result = processor.processResponse(searchRequest, emptyResponse, ctx);

        assertEquals("Processor should return the same response when there are no hits.", emptyResponse, result);
    }

    public void testProcessResponse_whenHappyCaseFloatWithL2_thenRerank() throws IOException {
        runProcessResponseRerankHappyCase(SpaceType.L2, VectorDataType.FLOAT);
    }

    public void testProcessResponse_whenHappyCaseBinaryWithHammingSpaceType_thenRerank() throws IOException {
        runProcessResponseRerankHappyCase(SpaceType.HAMMING, VectorDataType.BINARY);
    }

    private void runProcessResponseRerankHappyCase(SpaceType spaceType, VectorDataType vectorDataType) throws IOException {
        SearchResponse searchResponse = createSearchResponse();
        assertEquals(10, searchResponse.getInternalResponse().hits().getHits().length);
        assertEquals(0, searchResponse.getInternalResponse().hits().getHits()[0].docId());
        assertNotNull(
            "Should have the knn_vector in the source.",
            searchResponse.getInternalResponse().hits().getHits()[0].getSourceAsMap().get("knn_vector")
        );
        assertEquals(1, searchResponse.getInternalResponse().hits().getHits()[1].docId());
        assertEquals(2, searchResponse.getInternalResponse().hits().getHits()[2].docId());

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setDiversity(0.5f);
        mmrRerankContext.setOriginalQuerySize(3);
        mmrRerankContext.setSpaceType(spaceType);
        mmrRerankContext.setVectorDataType(vectorDataType);
        mmrRerankContext.setVectorFieldPath("knn_vector");
        mmrRerankContext.setOriginalFetchSourceContext(new FetchSourceContext(true, new String[] {}, new String[] { "knn_vector" }));
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        SearchResponse result = processor.processResponse(searchRequest, searchResponse, ctx);

        assertEquals("Should reduce the hits to the original query size.", 3, result.getInternalResponse().hits().getHits().length);
        assertEquals(0, result.getInternalResponse().hits().getHits()[0].docId());
        assertNull(
            "Should exclude the knn_vector from the source.",
            result.getInternalResponse().hits().getHits()[0].getSourceAsMap().get("knn_vector")
        );
        assertEquals("Should pick the hit with diversity.", 8, result.getInternalResponse().hits().getHits()[1].docId());
        assertEquals("Should pick the hit with diversity.", 9, result.getInternalResponse().hits().getHits()[2].docId());
    }

    public void testProcessResponse_whenMissingRerankContext_thenException() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        PipelineProcessingContext ctx = new PipelineProcessingContext();

        IllegalStateException exception = assertThrows(
            IllegalStateException.class,
            () -> processor.processResponse(searchRequest, searchResponse, ctx)
        );
        String expectedMessage = "MMR rerank context cannot be null";
        assertEquals(expectedMessage, exception.getMessage());
    }

    public void testProcessResponse_whenMissingSpaceType_thenException() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        IllegalStateException exception = assertThrows(
            IllegalStateException.class,
            () -> processor.processResponse(searchRequest, searchResponse, ctx)
        );
        String expectedMessage = "Space type in MMR rerank context cannot be null";
        assertEquals(expectedMessage, exception.getMessage());
    }

    public void testProcessResponse_whenMissingOriginalQuerySize_thenException() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setSpaceType(SpaceType.L2);
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        IllegalStateException exception = assertThrows(
            IllegalStateException.class,
            () -> processor.processResponse(searchRequest, searchResponse, ctx)
        );
        String expectedMessage = "Original query size in MMR rerank context cannot be null";
        assertEquals(expectedMessage, exception.getMessage());
    }

    @SuppressWarnings("unchecked")
    public void testProcessResponse_whenExplainEnabled_thenExplainInfoInSource() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setDiversity(0.5f);
        mmrRerankContext.setOriginalQuerySize(3);
        mmrRerankContext.setSpaceType(SpaceType.L2);
        mmrRerankContext.setVectorDataType(VectorDataType.FLOAT);
        mmrRerankContext.setVectorFieldPath("knn_vector");
        mmrRerankContext.setExplain(true);
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        SearchResponse result = processor.processResponse(searchRequest, searchResponse, ctx);

        assertEquals(3, result.getInternalResponse().hits().getHits().length);

        for (SearchHit hit : result.getInternalResponse().hits().getHits()) {
            Map<String, Object> source = hit.getSourceAsMap();
            assertNotNull("mmr_explain should be present when explain is enabled", source.get(MMR_EXPLAIN));
            Map<String, Object> explainInfo = (Map<String, Object>) source.get(MMR_EXPLAIN);
            assertNotNull("original_score should be present", explainInfo.get(MMRExplainInfo.ORIGINAL_SCORE_FIELD));
            assertNotNull("max_similarity_to_selected should be present", explainInfo.get(MMRExplainInfo.MAX_SIMILARITY_TO_SELECTED_FIELD));
            assertNotNull("mmr_score should be present", explainInfo.get(MMRExplainInfo.MMR_SCORE_FIELD));
            assertNotNull("mmr_formula should be present", explainInfo.get(MMRExplainInfo.MMR_FORMULA_FIELD));
            // selection_round and selected_so_far are intentionally omitted - they can be inferred from result position
            assertNull("selection_round should NOT be present (redundant with result position)", explainInfo.get("selection_round"));
            assertNull("selected_so_far should NOT be present (redundant with preceding hits)", explainInfo.get("selected_so_far"));
        }

        // Verify first hit has max_similarity_to_selected = 0 (no prior selections)
        Map<String, Object> firstExplain = (Map<String, Object>) result.getInternalResponse().hits().getHits()[0].getSourceAsMap()
            .get(MMR_EXPLAIN);
        assertEquals(0.0f, ((Number) firstExplain.get(MMRExplainInfo.MAX_SIMILARITY_TO_SELECTED_FIELD)).floatValue(), 1e-6);
    }

    public void testProcessResponse_whenExplainDisabled_thenNoExplainInfoInSource() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setDiversity(0.5f);
        mmrRerankContext.setOriginalQuerySize(3);
        mmrRerankContext.setSpaceType(SpaceType.L2);
        mmrRerankContext.setVectorDataType(VectorDataType.FLOAT);
        mmrRerankContext.setVectorFieldPath("knn_vector");
        mmrRerankContext.setExplain(false);
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        SearchResponse result = processor.processResponse(searchRequest, searchResponse, ctx);

        assertEquals(3, result.getInternalResponse().hits().getHits().length);

        for (SearchHit hit : result.getInternalResponse().hits().getHits()) {
            Map<String, Object> source = hit.getSourceAsMap();
            assertNull("mmr_explain should NOT be present when explain is disabled", source.get(MMR_EXPLAIN));
        }
    }

    public void testProcessResponse_whenExplainNull_thenNoExplainInfoInSource() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setDiversity(0.5f);
        mmrRerankContext.setOriginalQuerySize(3);
        mmrRerankContext.setSpaceType(SpaceType.L2);
        mmrRerankContext.setVectorDataType(VectorDataType.FLOAT);
        mmrRerankContext.setVectorFieldPath("knn_vector");
        // explain is not set (null)
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        SearchResponse result = processor.processResponse(searchRequest, searchResponse, ctx);

        assertEquals(3, result.getInternalResponse().hits().getHits().length);

        for (SearchHit hit : result.getInternalResponse().hits().getHits()) {
            Map<String, Object> source = hit.getSourceAsMap();
            assertNull("mmr_explain should NOT be present when explain is null", source.get(MMR_EXPLAIN));
        }
    }

    @SuppressWarnings("unchecked")
    public void testProcessResponse_whenExplainEnabledWithFetchSourceFilter_thenExplainInfoFilteredCorrectly() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setDiversity(0.5f);
        mmrRerankContext.setOriginalQuerySize(3);
        mmrRerankContext.setSpaceType(SpaceType.L2);
        mmrRerankContext.setVectorDataType(VectorDataType.FLOAT);
        mmrRerankContext.setVectorFieldPath("knn_vector");
        mmrRerankContext.setExplain(true);
        // Exclude knn_vector but include mmr_explain
        mmrRerankContext.setOriginalFetchSourceContext(new FetchSourceContext(true, new String[] {}, new String[] { "knn_vector" }));
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        SearchResponse result = processor.processResponse(searchRequest, searchResponse, ctx);

        assertEquals(3, result.getInternalResponse().hits().getHits().length);

        for (SearchHit hit : result.getInternalResponse().hits().getHits()) {
            Map<String, Object> source = hit.getSourceAsMap();
            // mmr_explain should still be present since it's not excluded
            assertNotNull("mmr_explain should be present", source.get(MMR_EXPLAIN));
            // knn_vector should be excluded
            assertNull("knn_vector should be excluded", source.get("knn_vector"));
        }
    }

    public void testProcessResponse_whenMissingDiversity_thenException() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setSpaceType(SpaceType.L2);
        mmrRerankContext.setOriginalQuerySize(3);
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        IllegalStateException exception = assertThrows(
            IllegalStateException.class,
            () -> processor.processResponse(searchRequest, searchResponse, ctx)
        );
        String expectedMessage = "Diversity in MMR rerank context cannot be null";
        assertEquals(expectedMessage, exception.getMessage());
    }

    private SearchResponse createSearchResponse() throws IOException {
        SearchHit[] hits = new SearchHit[10];

        // 8 similar hits, high score
        float[] similarVector = new float[] { 1f, 1f };
        for (int i = 0; i < 8; i++) {
            XContentBuilder sourceBuilder = JsonXContent.contentBuilder().startObject().array("knn_vector", similarVector).endObject();

            SearchHit hit = new SearchHit(i, String.valueOf(i), Map.of(), Map.of());
            hit.sourceRef(BytesReference.bytes(sourceBuilder));
            hit.score(1f);
            hits[i] = hit;
        }

        // 2 diverse hits, slightly lower score
        float[][] diverseVectors = new float[][] { { 1f, 2f }, { 2f, 1f } };
        for (int i = 0; i < 2; i++) {
            int idx = i + 8;
            XContentBuilder sourceBuilder = JsonXContent.contentBuilder().startObject().array("knn_vector", diverseVectors[i]).endObject();

            SearchHit hit = new SearchHit(idx, String.valueOf(idx), Map.of(), Map.of());
            hit.sourceRef(BytesReference.bytes(sourceBuilder));
            hit.score(0.8f); // slightly lower than top similar hits
            hits[idx] = hit;
        }
        return createSearchResponse(hits);
    }

    private SearchResponse createSearchResponse(SearchHit... hits) {
        TotalHits totalHits = new TotalHits(hits.length, TotalHits.Relation.EQUAL_TO);

        float maxScore = Arrays.stream(hits).map(SearchHit::getScore).max(Float::compare).orElse(Float.NEGATIVE_INFINITY);

        SearchHits searchHits = new SearchHits(hits, totalHits, maxScore);

        SearchResponseSections sections = new SearchResponseSections(
            searchHits,
            null,   // aggregations
            null,   // suggest
            false,  // timedOut
            false,  // terminatedEarly
            null,   // profileShardResults
            0       // numReducePhases
        );

        return new SearchResponse(
            sections,
            null,   // scrollId
            1,      // totalShards
            1,      // successfulShards
            0,      // skippedShards
            1,      // tookInMillis
            new ShardSearchFailure[0],
            new SearchResponse.Clusters(1, 1, 0),
            null    // pitId
        );
    }
}
