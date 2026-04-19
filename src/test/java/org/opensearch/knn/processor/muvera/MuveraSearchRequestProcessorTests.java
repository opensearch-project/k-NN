/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import org.opensearch.action.search.SearchRequest;
import org.opensearch.index.query.TemplateQueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.pipeline.PipelineProcessingContext;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MuveraSearchRequestProcessorTests extends KNNTestCase {

    private static final String TARGET_FIELD = "muvera_fde";
    private static final int DIM = 4;
    private static final int K_SIM = 1;
    private static final int DIM_PROJ = 2;
    private static final int R_REPS = 2;
    private static final int FDE_DIM = R_REPS * (1 << K_SIM) * DIM_PROJ; // 8
    private static final String QUERY_VECTORS_FIELD = "ext.muvera.query_vectors";

    private MuveraSearchRequestProcessor createProcessor() {
        MuveraEncoder encoder = new MuveraEncoder(DIM, K_SIM, DIM_PROJ, R_REPS, 42L);
        return new MuveraSearchRequestProcessor(
            "test_tag",
            "test description",
            false,
            TARGET_FIELD,
            encoder,
            DIM,
            FDE_DIM,
            QUERY_VECTORS_FIELD
        );
    }

    /**
     * Builds a template query content map that mirrors what a user would send:
     * <pre>
     * {
     *   "script_score": {
     *     "query": { "knn": { "muvera_fde": { "vector": "${muvera_fde}", "k": 10 } } },
     *     "script": {
     *       "source": "lateInteractionScore(...)",
     *       "params": { "query_vectors": [[...]], "space_type": "innerproduct" }
     *     }
     *   }
     * }
     * </pre>
     */
    private Map<String, Object> buildTemplateContent(List<List<Double>> queryVectors) {
        Map<String, Object> knnField = new HashMap<>();
        knnField.put("vector", "${" + TARGET_FIELD + "}");
        knnField.put("k", 10);

        Map<String, Object> knn = new HashMap<>();
        knn.put(TARGET_FIELD, knnField);

        Map<String, Object> innerQuery = new HashMap<>();
        innerQuery.put("knn", knn);

        Map<String, Object> params = new HashMap<>();
        params.put("query_vectors", queryVectors);
        params.put("space_type", "innerproduct");

        Map<String, Object> script = new HashMap<>();
        script.put("source", "lateInteractionScore(params.query_vectors, 'colbert_vectors', params._source, params.space_type)");
        script.put("params", params);

        Map<String, Object> scriptScore = new HashMap<>();
        scriptScore.put("query", innerQuery);
        scriptScore.put("script", script);

        Map<String, Object> content = new HashMap<>();
        content.put("script_score", scriptScore);
        return content;
    }

    private SearchRequest buildTemplateRequest(List<List<Double>> queryVectors) {
        TemplateQueryBuilder template = new TemplateQueryBuilder(buildTemplateContent(queryVectors));
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(template).size(5));
        return request;
    }

    @SuppressWarnings("unchecked")
    public void testProcessRequestSetsContextAttribute() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        List<List<Double>> queryVectors = Arrays.asList(
            Arrays.asList(1.0, 0.0, 0.0, 0.0),
            Arrays.asList(0.0, 1.0, 0.0, 0.0)
        );
        SearchRequest request = buildTemplateRequest(queryVectors);
        PipelineProcessingContext context = new PipelineProcessingContext();

        SearchRequest processed = processor.processRequest(request, context);

        // The request itself is unchanged — template resolution happens later via the context
        assertSame(request, processed);
        assertTrue(processed.source().query() instanceof TemplateQueryBuilder);

        // The FDE vector must be stored in the context under the target field name
        Object fde = context.getAttribute(TARGET_FIELD);
        assertNotNull("Context attribute [" + TARGET_FIELD + "] must be set", fde);
        assertTrue("Context attribute must be a List<Float>", fde instanceof List);
        List<Float> fdeList = (List<Float>) fde;
        assertEquals("FDE length must match configured fde_dimension", FDE_DIM, fdeList.size());
        for (Float v : fdeList) {
            assertNotNull(v);
            assertTrue(Float.isFinite(v));
        }
    }

    public void testProcessRequestPassesThroughNonTemplateQuery() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(new TermQueryBuilder("field", "value")));
        PipelineProcessingContext context = new PipelineProcessingContext();

        SearchRequest processed = processor.processRequest(request, context);

        // Non-template query: pass through untouched, no context attribute
        assertTrue(processed.source().query() instanceof TermQueryBuilder);
        assertNull(context.getAttribute(TARGET_FIELD));
    }

    public void testProcessRequestPassesThroughNullQuery() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder()); // no query
        PipelineProcessingContext context = new PipelineProcessingContext();

        SearchRequest processed = processor.processRequest(request, context);
        assertNull(processed.source().query());
        assertNull(context.getAttribute(TARGET_FIELD));
    }

    public void testProcessRequestPassesThroughNullSource() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        SearchRequest request = new SearchRequest("test-index"); // no source
        PipelineProcessingContext context = new PipelineProcessingContext();

        SearchRequest processed = processor.processRequest(request, context);
        assertSame(request, processed);
        assertNull(context.getAttribute(TARGET_FIELD));
    }

    public void testProcessRequestPassesThroughTemplateWithoutQueryVectors() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        // Template query but the script has no query_vectors param
        Map<String, Object> params = new HashMap<>();
        params.put("space_type", "innerproduct");

        Map<String, Object> script = new HashMap<>();
        script.put("source", "some_other_script(...)");
        script.put("params", params);

        Map<String, Object> scriptScore = new HashMap<>();
        scriptScore.put("query", Map.of("match_all", Map.of()));
        scriptScore.put("script", script);

        Map<String, Object> content = new HashMap<>();
        content.put("script_score", scriptScore);

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(new TemplateQueryBuilder(content)));
        PipelineProcessingContext context = new PipelineProcessingContext();

        SearchRequest processed = processor.processRequest(request, context);
        assertTrue(processed.source().query() instanceof TemplateQueryBuilder);
        assertNull(context.getAttribute(TARGET_FIELD));
    }

    public void testProcessRequestThrowsOnDimensionMismatch() {
        MuveraSearchRequestProcessor processor = createProcessor();

        // dim=3 but processor expects dim=4
        List<List<Double>> badVectors = Arrays.asList(Arrays.asList(1.0, 0.0, 0.0));
        SearchRequest request = buildTemplateRequest(badVectors);

        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> processor.processRequest(request, new PipelineProcessingContext())
        );
        assertTrue(e.getMessage().contains("dimension"));
        assertTrue(e.getMessage().contains("3"));
        assertTrue(e.getMessage().contains("4"));
    }

    public void testProcessRequestThrowsOnEmptyQueryVectors() {
        MuveraSearchRequestProcessor processor = createProcessor();

        SearchRequest request = buildTemplateRequest(Arrays.asList());

        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> processor.processRequest(request, new PipelineProcessingContext())
        );
        assertTrue(e.getMessage().contains("must not be empty"));
    }

    public void testProcessRequestThrowsOnNonListQueryVectors() {
        MuveraSearchRequestProcessor processor = createProcessor();

        Map<String, Object> params = new HashMap<>();
        params.put("query_vectors", "not-a-list");
        params.put("space_type", "innerproduct");

        Map<String, Object> script = new HashMap<>();
        script.put("source", "lateInteractionScore(...)");
        script.put("params", params);

        Map<String, Object> scriptScore = new HashMap<>();
        scriptScore.put("query", Map.of("match_all", Map.of()));
        scriptScore.put("script", script);

        Map<String, Object> content = new HashMap<>();
        content.put("script_score", scriptScore);

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(new TemplateQueryBuilder(content)));

        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> processor.processRequest(request, new PipelineProcessingContext())
        );
        assertTrue(e.getMessage().contains("must be a list of vectors"));
    }

    public void testProcessRequestWithoutContextDoesNotResolve() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        List<List<Double>> queryVectors = Arrays.asList(Arrays.asList(1.0, 0.0, 0.0, 0.0));
        SearchRequest request = buildTemplateRequest(queryVectors);

        // Overload without PipelineProcessingContext should be a no-op
        SearchRequest processed = processor.processRequest(request);
        assertSame(request, processed);
    }

    public void testGetType() {
        assertEquals("muvera_query", createProcessor().getType());
    }

    // ---------------------- Factory tests ----------------------

    public void testFactoryCreateWithDefaults() throws Exception {
        MuveraSearchRequestProcessor.Factory factory = new MuveraSearchRequestProcessor.Factory();

        Map<String, Object> config = new HashMap<>();
        config.put("target_field", TARGET_FIELD);
        config.put("dim", DIM);

        MuveraSearchRequestProcessor processor = factory.create(Map.of(), "tag", "desc", false, config, null);
        assertNotNull(processor);
        assertEquals("muvera_query", processor.getType());
    }

    public void testFactoryThrowsOnMissingDim() {
        MuveraSearchRequestProcessor.Factory factory = new MuveraSearchRequestProcessor.Factory();

        Map<String, Object> config = new HashMap<>();
        config.put("target_field", TARGET_FIELD);
        // dim is missing

        Exception e = expectThrows(Exception.class, () -> factory.create(Map.of(), "tag", "desc", false, config, null));
        assertTrue(e.getMessage().contains("dim"));
    }

    public void testFactoryThrowsOnFdeDimensionMismatch() {
        MuveraSearchRequestProcessor.Factory factory = new MuveraSearchRequestProcessor.Factory();

        Map<String, Object> config = new HashMap<>();
        config.put("target_field", TARGET_FIELD);
        config.put("dim", DIM);
        config.put("k_sim", K_SIM);
        config.put("dim_proj", DIM_PROJ);
        config.put("r_reps", R_REPS);
        config.put("fde_dimension", 999);

        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> factory.create(Map.of(), "tag", "desc", false, config, null)
        );
        assertTrue(e.getMessage().contains("fde_dimension"));
    }
}
