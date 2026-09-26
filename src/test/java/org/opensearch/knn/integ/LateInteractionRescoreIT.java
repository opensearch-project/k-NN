/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;
import static org.opensearch.knn.index.query.parser.RescoreParser.LATE_INTERACTION_VECTOR_PARAMETER;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_PARAMETER;

/**
 * End-to-end integration test for the native late-interaction (MaxSim) rescore path:
 * index docs with an FDE {@code knn_vector} field and a {@code late_interaction} multi-vector field,
 * run a {@code knn} query with a {@code late_interaction} rescore, and assert the results are reranked
 * by MaxSim.
 */
public class LateInteractionRescoreIT extends KNNRestTestCase {

    private static final String FDE_FIELD = "fde";
    private static final String LI_FIELD = "tokens";
    private static final int FDE_DIM = 2;
    private static final int LI_DIM = 2;

    @SneakyThrows
    public void testLateInteractionRescore_reranksByMaxSim() {
        createIndexWithLateInteractionField();

        // All three docs share a near-identical FDE so phase-1 kNN returns all of them; the ranking is
        // then decided by MaxSim over the token multi-vectors.
        // Query tokens: [[1,0]]. MaxSim = max over doc tokens of innerproduct with [1,0].
        // doc1 tokens [[1,0],[0,1]] -> max(1, 0) = 1.0 (winner)
        // doc2 tokens [[0.5,0],[0,1]] -> max(0.5,0) = 0.5
        // doc3 tokens [[0,1]] -> max(0) = 0.0
        indexDoc("doc1", new float[] { 1.0f, 0.0f }, new float[][] { { 1.0f, 0.0f }, { 0.0f, 1.0f } });
        indexDoc("doc2", new float[] { 1.0f, 0.01f }, new float[][] { { 0.5f, 0.0f }, { 0.0f, 1.0f } });
        indexDoc("doc3", new float[] { 1.0f, 0.02f }, new float[][] { { 0.0f, 1.0f } });
        refreshAllNonSystemIndices();

        final float[][] queryTokens = { { 1.0f, 0.0f } };
        final Response response = search(new float[] { 1.0f, 0.0f }, 3, LI_FIELD, queryTokens, 2.0f);
        assertEquals(RestStatus.OK.getStatus(), response.getStatusLine().getStatusCode());

        final String body = org.apache.hc.core5.http.io.entity.EntityUtils.toString(response.getEntity());
        final List<String> ids = hitIdsInOrder(body);
        assertEquals(3, ids.size());
        assertEquals("MaxSim winner should rank first", "doc1", ids.get(0));
        assertEquals("doc2 should rank second", "doc2", ids.get(1));
        assertEquals("doc3 (lowest MaxSim) should rank last", "doc3", ids.get(2));
    }

    @SneakyThrows
    public void testLateInteractionRescore_oversamplesPhase1Pool() {
        // Regression for the phase-1 oversampling wiring: the true MaxSim winner must be surfaced even
        // when the FDE prefetch does not rank it first. With k=1 and oversample_factor>=3, the phase-1
        // pool must include >=3 candidates so MaxSim can promote the real winner.
        createIndexWithLateInteractionField();

        // FDE nearest to query [1,0] is doc_a, then doc_b, then doc_c (by the 2nd coordinate).
        // But MaxSim(query=[1,0]) winner is doc_c (token [1,0] -> 1.0); doc_a/doc_b have max 0.2/0.5.
        indexDoc("doc_a", new float[] { 1.0f, 0.00f }, new float[][] { { 0.2f, 0.0f } });
        indexDoc("doc_b", new float[] { 1.0f, 0.01f }, new float[][] { { 0.5f, 0.0f } });
        indexDoc("doc_c", new float[] { 1.0f, 0.02f }, new float[][] { { 1.0f, 0.0f } });
        refreshAllNonSystemIndices();

        // k=1 but oversample 3 -> phase-1 returns 3 candidates -> MaxSim picks doc_c.
        final Request request = searchRequest(new float[] { 1.0f, 0.0f }, 1, LI_FIELD, new float[][] { { 1.0f, 0.0f } }, 3.0f);
        final Response response = client().performRequest(request);
        final String body = org.apache.hc.core5.http.io.entity.EntityUtils.toString(response.getEntity());
        final List<String> ids = hitIdsInOrder(body);
        assertEquals(1, ids.size());
        assertEquals("oversampling must let MaxSim surface the true winner past the FDE order", "doc_c", ids.get(0));
    }

    @SneakyThrows
    public void testLateInteractionRescore_missingField_thenFails() {
        createIndexWithLateInteractionField();
        indexDoc("doc1", new float[] { 1.0f, 0.0f }, new float[][] { { 1.0f, 0.0f } });
        refreshAllNonSystemIndices();

        // rescore targets a field that does not exist
        final Request request = searchRequest(new float[] { 1.0f, 0.0f }, 3, "no_such_field", new float[][] { { 1.0f, 0.0f } }, 2.0f);
        final ResponseException e = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertThat(org.apache.hc.core5.http.io.entity.EntityUtils.toString(e.getResponse().getEntity()), containsString("no_such_field"));
    }

    @SneakyThrows
    public void testLateInteractionRescore_dimensionMismatch_thenFails() {
        createIndexWithLateInteractionField();
        indexDoc("doc1", new float[] { 1.0f, 0.0f }, new float[][] { { 1.0f, 0.0f } });
        refreshAllNonSystemIndices();

        // query tokens have dimension 3, field expects 2
        final Request request = searchRequest(new float[] { 1.0f, 0.0f }, 3, LI_FIELD, new float[][] { { 1.0f, 0.0f, 0.0f } }, 2.0f);
        expectThrows(ResponseException.class, () -> client().performRequest(request));
    }

    // ---- helpers ----

    private void createIndexWithLateInteractionField() throws IOException {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FDE_FIELD)
            .field("type", "knn_vector")
            .field("dimension", FDE_DIM)
            .field("space_type", "innerproduct")
            .startObject("method")
            .field("name", "hnsw")
            .field("engine", "lucene")
            .endObject()
            .endObject()
            .startObject(LI_FIELD)
            .field("type", "late_interaction")
            .field("dimension", LI_DIM)
            .field("space_type", "innerproduct")
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, mapping.toString());
    }

    private void indexDoc(String id, float[] fde, float[][] tokens) throws IOException {
        XContentBuilder doc = XContentFactory.jsonBuilder().startObject();
        doc.field(FDE_FIELD, fde);
        doc.startArray(LI_FIELD);
        for (float[] t : tokens) {
            doc.startArray();
            for (float v : t) {
                doc.value(v);
            }
            doc.endArray();
        }
        doc.endArray();
        doc.endObject();
        addKnnDoc(INDEX_NAME, id, doc.toString());
    }

    private Response search(float[] fdeVector, int k, String liField, float[][] queryTokens, float oversample) throws IOException {
        return client().performRequest(searchRequest(fdeVector, k, liField, queryTokens, oversample));
    }

    private Request searchRequest(float[] fdeVector, int k, String liField, float[][] queryTokens, float oversample) throws IOException {
        // RFC #3439 DSL: rescore.{oversample_factor, <li_field>:{vector:[[...]]}}
        XContentBuilder b = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FDE_FIELD)
            .field("vector", fdeVector)
            .field("k", k)
            .startObject(RESCORE_PARAMETER)
            .field("oversample_factor", oversample)
            .startObject(liField)
            .startArray(LATE_INTERACTION_VECTOR_PARAMETER);
        for (float[] t : queryTokens) {
            b.startArray();
            for (float v : t) {
                b.value(v);
            }
            b.endArray();
        }
        b.endArray().endObject().endObject().endObject().endObject().endObject().endObject();

        Request request = new Request("POST", String.format(Locale.ROOT, "/%s/_search", INDEX_NAME));
        request.addParameter("size", Integer.toString(10));
        request.setJsonEntity(b.toString());
        return request;
    }

    @SuppressWarnings("unchecked")
    private List<String> hitIdsInOrder(String responseBody) throws IOException {
        final List<Object> hits = parseSearchResponseHits(responseBody);
        return hits.stream().map(h -> (String) ((Map<String, Object>) h).get("_id")).collect(java.util.stream.Collectors.toList());
    }
}
