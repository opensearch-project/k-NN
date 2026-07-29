/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.knn.KNNRestTestCase;

import java.io.IOException;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.hamcrest.Matchers.containsString;

/**
 * End-to-end tests for dynamic knn_vector mapping: the {@code match_mapping_type: "knn_vector"}
 * dynamic-template path (index-creation validation) and the auto-inference path (document indexing),
 * including a dimension sweep across the ≥128-and-multiple-of-8 gate.
 */
public class DynamicVectorMappingIT extends KNNRestTestCase {

    /** A JSON array of {@code n} floats. */
    private static String numericArray(int n) {
        return "[" + IntStream.range(0, n).mapToObj(i -> "0.1").collect(Collectors.joining(",")) + "]";
    }

    /** PUT an index whose only mapping is a knn_vector dynamic template with the given mapping block. */
    private Response putIndexWithKnnTemplate(String index, String templateMappingBlock) throws IOException {
        String body = "{"
            + "\"settings\": {\"index.knn\": true},"
            + "\"mappings\": {\"dynamic_templates\": [ {\"vectors\": {"
            + "\"match_mapping_type\": \"knn_vector\", \"mapping\": "
            + templateMappingBlock
            + "}} ]}}";
        Request request = new Request("PUT", "/" + index);
        request.setJsonEntity(body);
        return client().performRequest(request);
    }

    private String fieldType(String index, String field) throws IOException, ParseException {
        Request request = new Request("GET", "/" + index + "/_mapping");
        String resp = EntityUtils.toString(client().performRequest(request).getEntity());
        // crude but sufficient: look for "field":{"type":"..."} without pulling in a JSON dep here
        int f = resp.indexOf("\"" + field + "\"");
        if (f < 0) return null;
        int t = resp.indexOf("\"type\":\"", f);
        if (t < 0) return null;
        t += "\"type\":\"".length();
        return resp.substring(t, resp.indexOf("\"", t));
    }

    private void indexDoc(String index, String field, String jsonValue) throws IOException {
        Request request = new Request("POST", "/" + index + "/_doc?refresh=true");
        request.setJsonEntity("{\"" + field + "\": " + jsonValue + "}");
        client().performRequest(request);
    }

    // ---- Index-creation validation: COMPLETE config → eager TypeParser validation → FAIL ----

    public void testNegativeDimensionRejectedAtIndexCreation() throws IOException, ParseException {
        ResponseException e = expectThrows(
            ResponseException.class,
            () -> putIndexWithKnnTemplate("dv_neg", "{\"type\": \"knn_vector\", \"dimension\": -3}")
        );
        assertThat(EntityUtils.toString(e.getResponse().getEntity()), containsString("Dimension value must be greater than 0"));
    }

    public void testZeroDimensionRejectedAtIndexCreation() throws IOException, ParseException {
        ResponseException e = expectThrows(
            ResponseException.class,
            () -> putIndexWithKnnTemplate("dv_zero", "{\"type\": \"knn_vector\", \"dimension\": 0}")
        );
        assertThat(EntityUtils.toString(e.getResponse().getEntity()), containsString("Dimension value must be greater than 0"));
    }

    public void testNonNumericDimensionRejectedAtIndexCreation() {
        expectThrows(
            ResponseException.class,
            () -> putIndexWithKnnTemplate("dv_nan", "{\"type\": \"knn_vector\", \"dimension\": \"abc\"}")
        );
    }

    public void testTooLargeDimensionRejectedAtIndexCreation() {
        expectThrows(ResponseException.class, () -> putIndexWithKnnTemplate("dv_big", "{\"type\": \"knn_vector\", \"dimension\": 100000}"));
    }

    // ---- Index-creation validation: INCOMPLETE/valid config → accepted (deferred or valid) ----

    public void testValidDimensionAcceptedAtIndexCreation() throws IOException {
        putIndexWithKnnTemplate("dv_ok", "{\"type\": \"knn_vector\", \"dimension\": 128}");
        deleteKNNIndex("dv_ok");
    }

    public void testNoDimensionDeferredAndAccepted() throws IOException {
        putIndexWithKnnTemplate("dv_defer", "{\"type\": \"knn_vector\"}");
        deleteKNNIndex("dv_defer");
    }

    public void testEmptyMappingBlockDeferredAndAccepted() throws IOException {
        putIndexWithKnnTemplate("dv_empty", "{}");
        deleteKNNIndex("dv_empty");
    }

    public void testNamePlaceholderSkipsEagerValidation() throws IOException {
        // {name} can't be resolved up front, so even a config that would otherwise be validated is deferred.
        putIndexWithKnnTemplate("dv_name", "{\"type\": \"knn_vector\", \"field_name\": \"{name}\"}");
        deleteKNNIndex("dv_name");
    }

    // ---- Template path: dimension inferred from first doc, then locked ----

    public void testTemplateInfersDimensionFromFirstDoc() throws IOException, ParseException {
        putIndexWithKnnTemplate("dv_infer", "{\"type\": \"knn_vector\"}");
        indexDoc("dv_infer", "emb", numericArray(384));
        assertEquals("knn_vector", fieldType("dv_infer", "emb"));
        deleteKNNIndex("dv_infer");
    }

    public void testTemplateDimensionMismatchRejectedOnSecondDoc() throws IOException, ParseException {
        putIndexWithKnnTemplate("dv_lock", "{\"type\": \"knn_vector\"}");
        indexDoc("dv_lock", "emb", numericArray(128)); // locks dimension to 128
        ResponseException e = expectThrows(ResponseException.class, () -> indexDoc("dv_lock", "emb", numericArray(256)));
        assertThat(EntityUtils.toString(e.getResponse().getEntity()), containsString("dimension"));
        deleteKNNIndex("dv_lock");
    }

    // ---- Auto-inference path (no template): dimension sweep across the gate ----

    public void testAutoInferenceDimensionSweep() throws IOException, ParseException {
        createIndex("dv_sweep", getKNNDefaultIndexSettings());
        int min = 128;
        int[] dims = { 8, 64, 120, 127, 128, 129, 130, 136, 200, 256, 384, 512, 768, 1024 };
        for (int d : dims) {
            String field = "f" + d;
            indexDoc("dv_sweep", field, numericArray(d));
            String type = fieldType("dv_sweep", field);
            boolean expectKnn = d >= min && d % 8 == 0;
            if (expectKnn) {
                assertEquals(String.format(Locale.ROOT, "dim %d must infer knn_vector", d), "knn_vector", type);
            } else {
                assertNotEquals(String.format(Locale.ROOT, "dim %d must NOT infer knn_vector", d), "knn_vector", type);
            }
        }
        deleteKNNIndex("dv_sweep");
    }
}
