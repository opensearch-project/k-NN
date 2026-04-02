/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.KNNRestTestCase;

import java.io.IOException;
import java.util.List;

/**
 * Integration tests for MUVERA ingest and search request processors.
 *
 * Tests the end-to-end flow:
 * 1. Create ingest pipeline with muvera processor to encode multi-vectors into FDE at index time
 * 2. Create search pipeline with muvera_query processor to encode query multi-vectors and replace
 *    match_all with knn query for ANN prefetch
 * 3. Index documents with multi-vectors through the ingest pipeline
 * 4. Search using script_score with lateInteractionScore through the search pipeline
 */
public class MuveraProcessorIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "muvera-test-index";
    private static final String INGEST_PIPELINE_NAME = "muvera-ingest-pipeline";
    private static final String SEARCH_PIPELINE_NAME = "muvera-search-pipeline";
    private static final String MULTI_VECTOR_FIELD = "colbert_vectors";
    private static final String FDE_FIELD = "muvera_fde";

    // Use small MUVERA params for fast tests: FDE dim = 2 * 2 * 2 = 8
    private static final int DIM = 2;
    private static final int K_SIM = 1;
    private static final int DIM_PROJ = 2;
    private static final int R_REPS = 2;
    private static final int FDE_DIM = R_REPS * (1 << K_SIM) * DIM_PROJ; // 8
    private static final long SEED = 42L;

    /**
     * Tests the full MUVERA ingest + search pipeline flow end-to-end.
     * Creates pipelines, indexes documents with multi-vectors, and searches
     * using script_score with lateInteractionScore reranking over ANN prefetch.
     */
    public void testMuveraEndToEnd_whenIngestAndSearch_thenReturnsResults() throws Exception {
        try {
            createIngestPipeline();
            createSearchPipeline();
            createIndex();
            indexDocuments();
            refreshIndex(INDEX_NAME);

            // Search with script_score + lateInteractionScore through the search pipeline
            String searchBody = buildMuveraSearchBody(new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 } }, 5);
            Response response = performSearchWithPipeline(INDEX_NAME, searchBody, SEARCH_PIPELINE_NAME);
            assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

            String responseBody = EntityUtils.toString(response.getEntity());
            assertTrue("Response should contain hits", responseBody.contains("\"hits\""));

            List<Double> scores = parseScores(responseBody);
            assertFalse("Should have at least one result", scores.isEmpty());
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
            deleteSearchPipeline(SEARCH_PIPELINE_NAME);
        }
    }

    /**
     * Tests that the ingest pipeline correctly produces an FDE field on indexed documents.
     */
    public void testMuveraIngest_whenDocumentIndexed_thenFdeFieldPresent() throws Exception {
        try {
            createIngestPipeline();
            createIndex();
            indexDocuments();
            refreshIndex(INDEX_NAME);

            // Retrieve doc and verify FDE field exists
            Response getResponse = client().performRequest(new Request("GET", "/" + INDEX_NAME + "/_doc/1"));
            String body = EntityUtils.toString(getResponse.getEntity());
            assertTrue("Document should contain FDE field", body.contains(FDE_FIELD));
            // Original multi-vector field should also be preserved
            assertTrue("Document should preserve multi-vector field", body.contains(MULTI_VECTOR_FIELD));
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
        }
    }

    /**
     * Tests that creating an ingest pipeline with mismatched fde_dimension fails.
     */
    public void testMuveraIngest_whenFdeDimensionMismatch_thenFails() throws Exception {
        String pipelineBody = "{"
            + "\"description\": \"test muvera pipeline\","
            + "\"processors\": [{"
            + "  \"muvera\": {"
            + "    \"source_field\": \""
            + MULTI_VECTOR_FIELD
            + "\","
            + "    \"target_field\": \""
            + FDE_FIELD
            + "\","
            + "    \"dim\": "
            + DIM
            + ","
            + "    \"k_sim\": "
            + K_SIM
            + ","
            + "    \"dim_proj\": "
            + DIM_PROJ
            + ","
            + "    \"r_reps\": "
            + R_REPS
            + ","
            + "    \"seed\": "
            + SEED
            + ","
            + "    \"fde_dimension\": 999"
            + "  }"
            + "}]"
            + "}";

        Request request = new Request("PUT", "/_ingest/pipeline/" + INGEST_PIPELINE_NAME);
        request.setJsonEntity(pipelineBody);
        ResponseException e = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(
            "Error should mention fde_dimension mismatch",
            EntityUtils.toString(e.getResponse().getEntity()).contains("fde_dimension")
        );
    }

    /**
     * Tests that creating a search pipeline with mismatched fde_dimension fails.
     */
    public void testMuveraSearch_whenFdeDimensionMismatch_thenFails() throws Exception {
        String pipelineBody = "{"
            + "\"request_processors\": [{"
            + "  \"muvera_query\": {"
            + "    \"target_field\": \""
            + FDE_FIELD
            + "\","
            + "    \"dim\": "
            + DIM
            + ","
            + "    \"k_sim\": "
            + K_SIM
            + ","
            + "    \"dim_proj\": "
            + DIM_PROJ
            + ","
            + "    \"r_reps\": "
            + R_REPS
            + ","
            + "    \"seed\": "
            + SEED
            + ","
            + "    \"fde_dimension\": 999"
            + "  }"
            + "}]"
            + "}";

        Request request = new Request("PUT", "/_search/pipeline/" + SEARCH_PIPELINE_NAME);
        request.setJsonEntity(pipelineBody);
        ResponseException e = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(
            "Error should mention fde_dimension mismatch",
            EntityUtils.toString(e.getResponse().getEntity()).contains("fde_dimension")
        );
    }

    /**
     * Tests that the search pipeline passes through non-script_score queries unchanged.
     */
    public void testMuveraSearch_whenNonScriptScoreQuery_thenPassesThrough() throws Exception {
        try {
            createIngestPipeline();
            createSearchPipeline();
            createIndex();
            indexDocuments();
            refreshIndex(INDEX_NAME);

            // Simple match_all query (not script_score) through the MUVERA search pipeline
            String searchBody = "{\"query\": {\"match_all\": {}}, \"size\": 5}";
            Response response = performSearchWithPipeline(INDEX_NAME, searchBody, SEARCH_PIPELINE_NAME);
            assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

            String responseBody = EntityUtils.toString(response.getEntity());
            List<Double> scores = parseScores(responseBody);
            assertFalse("match_all should return results", scores.isEmpty());
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
            deleteSearchPipeline(SEARCH_PIPELINE_NAME);
        }
    }

    /**
     * Tests that search with dimension-mismatched query vectors fails with a clear error.
     */
    public void testMuveraSearch_whenQueryVectorDimensionMismatch_thenFails() throws Exception {
        try {
            createIngestPipeline();
            createSearchPipeline();
            createIndex();
            indexDocuments();
            refreshIndex(INDEX_NAME);

            // Query vectors with dim=3 but processor expects dim=2
            String searchBody = buildMuveraSearchBody(new double[][] { { 1.0, 0.0, 0.5 } }, 5);
            ResponseException e = expectThrows(
                ResponseException.class,
                () -> performSearchWithPipeline(INDEX_NAME, searchBody, SEARCH_PIPELINE_NAME)
            );
            String errorBody = EntityUtils.toString(e.getResponse().getEntity());
            assertTrue("Error should mention dimension mismatch", errorBody.contains("dimension"));
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
            deleteSearchPipeline(SEARCH_PIPELINE_NAME);
        }
    }

    /**
     * Tests that search pipeline with ignore_failure=true passes through on malformed query.
     */
    public void testMuveraSearch_whenIgnoreFailureTrue_thenMalformedQueryPassesThrough() throws Exception {
        String pipelineName = SEARCH_PIPELINE_NAME + "-ignore-failure";
        try {
            createIngestPipeline();
            createIndex();
            indexDocuments();
            refreshIndex(INDEX_NAME);

            // Create search pipeline with ignore_failure=true
            String pipelineBody = "{"
                + "\"request_processors\": [{"
                + "  \"muvera_query\": {"
                + "    \"target_field\": \"" + FDE_FIELD + "\","
                + "    \"dim\": " + DIM + ","
                + "    \"k_sim\": " + K_SIM + ","
                + "    \"dim_proj\": " + DIM_PROJ + ","
                + "    \"r_reps\": " + R_REPS + ","
                + "    \"seed\": " + SEED + ","
                + "    \"oversample_factor\": 2,"
                + "    \"ignore_failure\": true"
                + "  }"
                + "}]"
                + "}";
            Request pipelineRequest = new Request("PUT", "/_search/pipeline/" + pipelineName);
            pipelineRequest.setJsonEntity(pipelineBody);
            client().performRequest(pipelineRequest);

            // Query with wrong dimension — should not fail because ignore_failure=true
            String searchBody = buildMuveraSearchBody(new double[][] { { 1.0, 0.0, 0.5 } }, 5);
            Response response = performSearchWithPipeline(INDEX_NAME, searchBody, pipelineName);
            assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
            deleteSearchPipeline(pipelineName);
        }
    }

    /**
     * Tests that ingest pipeline with ignore_missing=false rejects documents without source field.
     */
    public void testMuveraIngest_whenIgnoreMissingFalse_thenMissingSourceFails() throws Exception {
        try {
            createIngestPipeline(); // default ignore_missing=false
            createIndex();

            // Index a document without the source field
            String docBody = "{\"text\": \"no vectors\"}";
            Request request = new Request("POST", "/" + INDEX_NAME + "/_doc/no_vectors?pipeline=" + INGEST_PIPELINE_NAME);
            request.setJsonEntity(docBody);
            ResponseException e = expectThrows(ResponseException.class, () -> client().performRequest(request));
            String errorBody = EntityUtils.toString(e.getResponse().getEntity());
            assertTrue("Error should mention missing field", errorBody.contains("not present"));
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
        }
    }

    /**
     * Tests that MUVERA search returns results in correct order based on MaxSim scoring.
     * Indexes multiple docs with known vectors, queries with a vector that clearly matches
     * one doc better, and verifies the top result is the expected doc.
     */
    public void testMuveraEndToEnd_whenSearchWithKnownVectors_thenOrderingIsCorrect() throws Exception {
        try {
            createIngestPipeline();
            createSearchPipeline();
            createIndex();

            // Doc "close": vectors very similar to query
            indexDocWithMultiVectors("close", new double[][] { { 0.9, 0.1 }, { 0.1, 0.9 } });
            // Doc "medium": somewhat similar
            indexDocWithMultiVectors("medium", new double[][] { { 0.5, 0.5 }, { 0.5, -0.5 } });
            // Doc "far": dissimilar
            indexDocWithMultiVectors("far", new double[][] { { -0.9, -0.1 }, { -0.1, -0.9 } });
            refreshIndex(INDEX_NAME);

            // Query with vectors very similar to "close" doc
            String searchBody = buildMuveraSearchBody(new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 } }, 3);
            Response response = performSearchWithPipeline(INDEX_NAME, searchBody, SEARCH_PIPELINE_NAME);
            assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

            String responseBody = EntityUtils.toString(response.getEntity());
            List<Double> scores = parseScores(responseBody);
            assertEquals("Should return all 3 docs", 3, scores.size());

            // Verify scores are in descending order
            for (int i = 0; i < scores.size() - 1; i++) {
                assertTrue(
                    "Scores should be in descending order: " + scores,
                    scores.get(i) >= scores.get(i + 1)
                );
            }

            // Verify the "close" doc has the highest score (appears first)
            assertTrue("Top result should have positive score", scores.get(0) > 0);
            assertTrue("Top score should be higher than second", scores.get(0) > scores.get(1));
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
            deleteSearchPipeline(SEARCH_PIPELINE_NAME);
        }
    }

    /**
     * Tests that MUVERA pipeline with ignore_missing=true handles mixed documents correctly.
     * One document has the source field, another doesn't.
     */
    public void testMuveraIngest_whenIgnoreMissingTrue_thenMixedDocsSucceed() throws Exception {
        try {
            // Create pipeline with ignore_missing=true
            String pipelineBody = "{"
                + "\"description\": \"MUVERA ingest pipeline with ignore_missing\","
                + "\"processors\": [{"
                + "  \"muvera\": {"
                + "    \"source_field\": \"" + MULTI_VECTOR_FIELD + "\","
                + "    \"target_field\": \"" + FDE_FIELD + "\","
                + "    \"dim\": " + DIM + ","
                + "    \"k_sim\": " + K_SIM + ","
                + "    \"dim_proj\": " + DIM_PROJ + ","
                + "    \"r_reps\": " + R_REPS + ","
                + "    \"ignore_missing\": true"
                + "  }"
                + "}]"
                + "}";
            Request pipelineRequest = new Request("PUT", "/_ingest/pipeline/" + INGEST_PIPELINE_NAME);
            pipelineRequest.setJsonEntity(pipelineBody);
            client().performRequest(pipelineRequest);

            createIndex();

            // Doc 1: has source field — should get FDE
            indexDocWithMultiVectors("with_vectors", new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 } });

            // Doc 2: missing source field — should be indexed without FDE
            String docBody = "{\"text\": \"no vectors here\"}";
            Request request = new Request("POST", "/" + INDEX_NAME + "/_doc/without_vectors?pipeline=" + INGEST_PIPELINE_NAME);
            request.setJsonEntity(docBody);
            Response response = client().performRequest(request);
            assertEquals(RestStatus.CREATED, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

            refreshIndex(INDEX_NAME);

            // Verify doc count
            Response countResponse = client().performRequest(new Request("GET", "/" + INDEX_NAME + "/_count"));
            String countBody = EntityUtils.toString(countResponse.getEntity());
            assertTrue("Should have 2 docs", countBody.contains("\"count\":2"));
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
        }
    }

    /**
     * Tests that the ingest pipeline rejects documents with wrong vector dimensions.
     */
    public void testMuveraIngest_whenVectorDimensionMismatch_thenFails() throws Exception {

        try {
            createIngestPipeline();
            createIndex();

            // Index a document with dim=3 vectors but processor expects dim=2
            String docBody = "{" + "\"" + MULTI_VECTOR_FIELD + "\": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]" + "}";
            Request request = new Request("POST", "/" + INDEX_NAME + "/_doc/bad_doc?pipeline=" + INGEST_PIPELINE_NAME);
            request.setJsonEntity(docBody);
            ResponseException e = expectThrows(ResponseException.class, () -> client().performRequest(request));
            String errorBody = EntityUtils.toString(e.getResponse().getEntity());
            assertTrue("Error should mention dimension mismatch", errorBody.contains("dimension"));
        } finally {
            deleteTestIndex(INDEX_NAME);
            deleteIngestPipeline(INGEST_PIPELINE_NAME);
        }
    }

    // ---- Helper methods ----

    private void createIngestPipeline() throws IOException {
        String pipelineBody = "{"
            + "\"description\": \"MUVERA ingest pipeline for IT\","
            + "\"processors\": [{"
            + "  \"muvera\": {"
            + "    \"source_field\": \""
            + MULTI_VECTOR_FIELD
            + "\","
            + "    \"target_field\": \""
            + FDE_FIELD
            + "\","
            + "    \"dim\": "
            + DIM
            + ","
            + "    \"k_sim\": "
            + K_SIM
            + ","
            + "    \"dim_proj\": "
            + DIM_PROJ
            + ","
            + "    \"r_reps\": "
            + R_REPS
            + ","
            + "    \"seed\": "
            + SEED
            + "  }"
            + "}]"
            + "}";

        Request request = new Request("PUT", "/_ingest/pipeline/" + INGEST_PIPELINE_NAME);
        request.setJsonEntity(pipelineBody);
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    private void createSearchPipeline() throws IOException {
        String pipelineBody = "{"
            + "\"request_processors\": [{"
            + "  \"muvera_query\": {"
            + "    \"target_field\": \""
            + FDE_FIELD
            + "\","
            + "    \"dim\": "
            + DIM
            + ","
            + "    \"k_sim\": "
            + K_SIM
            + ","
            + "    \"dim_proj\": "
            + DIM_PROJ
            + ","
            + "    \"r_reps\": "
            + R_REPS
            + ","
            + "    \"seed\": "
            + SEED
            + ","
            + "    \"oversample_factor\": 2"
            + "  }"
            + "}]"
            + "}";

        Request request = new Request("PUT", "/_search/pipeline/" + SEARCH_PIPELINE_NAME);
        request.setJsonEntity(pipelineBody);
        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    private void createIndex() throws IOException {
        String mapping = "{"
            + "\"properties\": {"
            + "  \""
            + FDE_FIELD
            + "\": {"
            + "    \"type\": \"knn_vector\","
            + "    \"dimension\": "
            + FDE_DIM
            + "  },"
            + "  \""
            + MULTI_VECTOR_FIELD
            + "\": {"
            + "    \"type\": \"object\","
            + "    \"enabled\": false"
            + "  }"
            + "}"
            + "}";
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void indexDocuments() throws IOException {
        // Document 1: two token vectors
        indexDocWithMultiVectors("1", new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 } });
        // Document 2: two token vectors (different)
        indexDocWithMultiVectors("2", new double[][] { { 0.5, 0.5 }, { -0.5, 0.5 } });
        // Document 3: single token vector
        indexDocWithMultiVectors("3", new double[][] { { 0.3, 0.7 } });
    }

    private void indexDocWithMultiVectors(String docId, double[][] vectors) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"").append(MULTI_VECTOR_FIELD).append("\": [");
        for (int i = 0; i < vectors.length; i++) {
            if (i > 0) sb.append(",");
            sb.append("[");
            for (int j = 0; j < vectors[i].length; j++) {
                if (j > 0) sb.append(",");
                sb.append(vectors[i][j]);
            }
            sb.append("]");
        }
        sb.append("]}");

        Request request = new Request("POST", "/" + INDEX_NAME + "/_doc/" + docId + "?pipeline=" + INGEST_PIPELINE_NAME + "&refresh=true");
        request.setJsonEntity(sb.toString());
        Response response = client().performRequest(request);
        assertEquals(RestStatus.CREATED, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    private String buildMuveraSearchBody(double[][] queryVectors, int size) {
        StringBuilder qvBuilder = new StringBuilder("[");
        for (int i = 0; i < queryVectors.length; i++) {
            if (i > 0) qvBuilder.append(",");
            qvBuilder.append("[");
            for (int j = 0; j < queryVectors[i].length; j++) {
                if (j > 0) qvBuilder.append(",");
                qvBuilder.append(queryVectors[i][j]);
            }
            qvBuilder.append("]");
        }
        qvBuilder.append("]");

        return "{"
            + "\"size\": "
            + size
            + ","
            + "\"query\": {"
            + "  \"script_score\": {"
            + "    \"query\": {\"match_all\": {}},"
            + "    \"script\": {"
            + "      \"source\": \"lateInteractionScore(params.query_vectors, '"
            + MULTI_VECTOR_FIELD
            + "', params._source)\","
            + "      \"params\": {"
            + "        \"query_vectors\": "
            + qvBuilder.toString()
            + "      }"
            + "    }"
            + "  }"
            + "}"
            + "}";
    }

    private Response performSearchWithPipeline(String index, String body, String pipelineName) throws IOException {
        Request request = new Request("POST", "/" + index + "/_search?search_pipeline=" + pipelineName);
        request.setJsonEntity(body);
        return client().performRequest(request);
    }

    private void deleteTestIndex(String index) {
        try {
            deleteKNNIndex(index);
        } catch (Exception e) {
            // Ignore — index may not exist
        }
    }

    private void deleteIngestPipeline(String pipelineId) {
        try {
            Request request = new Request("DELETE", "/_ingest/pipeline/" + pipelineId);
            client().performRequest(request);
        } catch (Exception e) {
            // Ignore — pipeline may not exist
        }
    }

    private void deleteSearchPipeline(String pipelineId) {
        try {
            Request request = new Request("DELETE", "/_search/pipeline/" + pipelineId);
            client().performRequest(request);
        } catch (Exception e) {
            // Ignore — pipeline may not exist
        }
    }
}
