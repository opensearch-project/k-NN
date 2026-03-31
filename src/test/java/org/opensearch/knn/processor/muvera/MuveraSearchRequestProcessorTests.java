/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import org.opensearch.action.search.SearchRequest;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptType;
import org.opensearch.search.builder.SearchSourceBuilder;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MuveraSearchRequestProcessorTests extends KNNTestCase {

    private static final String TARGET_FIELD = "muvera_fde";
    private static final int DIM = 4;
    private static final int K_SIM = 1;
    private static final int DIM_PROJ = 2;
    private static final int R_REPS = 2;
    private static final int FDE_DIM = R_REPS * (1 << K_SIM) * DIM_PROJ; // 8
    private static final int OVERSAMPLE = 4;

    private MuveraSearchRequestProcessor createProcessor() {
        MuveraEncoder encoder = new MuveraEncoder(DIM, K_SIM, DIM_PROJ, R_REPS, 42L);
        return new MuveraSearchRequestProcessor("test_tag", "test description", false, TARGET_FIELD, encoder, DIM, FDE_DIM, OVERSAMPLE);
    }

    public void testExtractScript() throws Exception {
        Map<String, Object> params = new HashMap<>();
        params.put("query_vectors", Arrays.asList(Arrays.asList(1.0, 0.0, 0.0, 0.0)));
        params.put("space_type", "innerproduct");

        Script script = new Script(
            ScriptType.INLINE,
            "painless",
            "lateInteractionScore(params.query_vectors, 'field', params._source, params.space_type)",
            params
        );
        ScriptScoreQueryBuilder ssq = new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script);

        Script extracted = MuveraSearchRequestProcessor.extractScript(ssq);
        assertNotNull(extracted);
        assertEquals(script.getIdOrCode(), extracted.getIdOrCode());
        assertNotNull(extracted.getParams());
        assertTrue(extracted.getParams().containsKey("query_vectors"));
        assertTrue(extracted.getParams().containsKey("space_type"));
    }

    public void testProcessRequestReplacesInnerQueryWithKnn() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        Map<String, Object> params = new HashMap<>();
        params.put("query_vectors", Arrays.asList(Arrays.asList(1.0, 0.0, 0.0, 0.0), Arrays.asList(0.0, 1.0, 0.0, 0.0)));
        params.put("space_type", "innerproduct");

        Script script = new Script(ScriptType.INLINE, "painless", "lateInteractionScore(...)", params);
        ScriptScoreQueryBuilder ssq = new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script);

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(ssq).size(5));

        SearchRequest processed = processor.processRequest(request);

        // The query should still be a ScriptScoreQueryBuilder
        QueryBuilder resultQuery = processed.source().query();
        assertTrue(resultQuery instanceof ScriptScoreQueryBuilder);

        // The inner query should now be a KNNQueryBuilder (not match_all)
        ScriptScoreQueryBuilder resultSsq = (ScriptScoreQueryBuilder) resultQuery;
        QueryBuilder innerQuery = resultSsq.query();
        assertTrue(
            "Inner query should be KNNQueryBuilder, got: " + innerQuery.getClass().getSimpleName(),
            innerQuery instanceof KNNQueryBuilder
        );
    }

    public void testProcessRequestPassesThroughNonScriptScore() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        // A simple term query — not script_score
        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(new TermQueryBuilder("field", "value")));

        SearchRequest processed = processor.processRequest(request);

        // Should pass through unchanged
        assertTrue(processed.source().query() instanceof TermQueryBuilder);
    }

    public void testProcessRequestPassesThroughWithoutQueryVectors() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        // script_score but no query_vectors in params
        Map<String, Object> params = new HashMap<>();
        params.put("field", "my_vector");

        Script script = new Script(ScriptType.INLINE, "painless", "knn_score(...)", params);
        ScriptScoreQueryBuilder ssq = new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script);

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(ssq));

        SearchRequest processed = processor.processRequest(request);

        // Should pass through unchanged — inner query still match_all
        ScriptScoreQueryBuilder resultSsq = (ScriptScoreQueryBuilder) processed.source().query();
        assertTrue(resultSsq.query() instanceof MatchAllQueryBuilder);
    }

    public void testProcessRequestPassesThroughNullQuery() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder()); // source with no query

        SearchRequest processed = processor.processRequest(request);
        assertNull(processed.source().query());
    }

    public void testProcessRequestThrowsOnDimensionMismatch() {
        MuveraSearchRequestProcessor processor = createProcessor();

        Map<String, Object> params = new HashMap<>();
        params.put(
            "query_vectors",
            Arrays.asList(
                Arrays.asList(1.0, 0.0, 0.0) // dim=3, expected dim=4
            )
        );

        Script script = new Script(ScriptType.INLINE, "painless", "lateInteractionScore(...)", params);
        ScriptScoreQueryBuilder ssq = new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script);

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(ssq));

        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> processor.processRequest(request));
        assertTrue(e.getMessage().contains("dimension"));
        assertTrue(e.getMessage().contains("3"));
        assertTrue(e.getMessage().contains("4"));
    }

    public void testProcessRequestThrowsOnEmptyQueryVectors() {
        MuveraSearchRequestProcessor processor = createProcessor();

        Map<String, Object> params = new HashMap<>();
        params.put("query_vectors", Arrays.asList()); // empty

        Script script = new Script(ScriptType.INLINE, "painless", "lateInteractionScore(...)", params);
        ScriptScoreQueryBuilder ssq = new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script);

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(ssq));

        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> processor.processRequest(request));
        assertTrue(e.getMessage().contains("must not be empty"));
    }

    public void testProcessRequestPreservesBoostAndMinScore() throws Exception {
        MuveraSearchRequestProcessor processor = createProcessor();

        Map<String, Object> params = new HashMap<>();
        params.put("query_vectors", Arrays.asList(Arrays.asList(1.0, 0.0, 0.0, 0.0)));

        Script script = new Script(ScriptType.INLINE, "painless", "lateInteractionScore(...)", params);
        ScriptScoreQueryBuilder ssq = new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script);
        ssq.boost(2.5f);
        ssq.setMinScore(1.0f);
        ssq.queryName("my_query");

        SearchRequest request = new SearchRequest("test-index");
        request.source(new SearchSourceBuilder().query(ssq).size(5));

        SearchRequest processed = processor.processRequest(request);
        ScriptScoreQueryBuilder resultSsq = (ScriptScoreQueryBuilder) processed.source().query();

        assertEquals(2.5f, resultSsq.boost(), 0.001f);
        assertEquals(1.0f, resultSsq.getMinScore(), 0.001f);
        assertEquals("my_query", resultSsq.queryName());
    }

    public void testGetType() {
        MuveraSearchRequestProcessor processor = createProcessor();
        assertEquals("muvera_query", processor.getType());
    }

    // Factory tests

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

    public void testFactoryThrowsOnInvalidOversampleFactor() {
        MuveraSearchRequestProcessor.Factory factory = new MuveraSearchRequestProcessor.Factory();

        Map<String, Object> config = new HashMap<>();
        config.put("target_field", TARGET_FIELD);
        config.put("dim", DIM);
        config.put("k_sim", K_SIM);
        config.put("dim_proj", DIM_PROJ);
        config.put("r_reps", R_REPS);
        config.put("oversample_factor", 0);

        Exception e = expectThrows(Exception.class, () -> factory.create(Map.of(), "tag", "desc", false, config, null));
        assertTrue(e.getMessage().contains("oversample_factor"));

        Map<String, Object> negativeConfig = new HashMap<>();
        negativeConfig.put("target_field", TARGET_FIELD);
        negativeConfig.put("dim", DIM);
        negativeConfig.put("k_sim", K_SIM);
        negativeConfig.put("dim_proj", DIM_PROJ);
        negativeConfig.put("r_reps", R_REPS);
        negativeConfig.put("oversample_factor", -1);

        Exception e2 = expectThrows(Exception.class, () -> factory.create(Map.of(), "tag", "desc", false, negativeConfig, null));
        assertTrue(e2.getMessage().contains("oversample_factor"));
    }
}
