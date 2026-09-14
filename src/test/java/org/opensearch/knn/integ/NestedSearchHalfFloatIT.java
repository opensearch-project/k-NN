/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;

import java.util.List;

public class NestedSearchHalfFloatIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "half_float_nested_test_index";
    private static final String NESTED_FIELD_NAME = "test_nested";
    private static final String VECTOR_FIELD_NAME = "test_vector";
    private static final String NESTED_VECTOR_PATH = NESTED_FIELD_NAME + "." + VECTOR_FIELD_NAME;
    private static final int DIMENSION = 4;

    @SneakyThrows
    public void testNestedSearch_whenHalfFloatFlat_thenReturnsExpectedResults() {
        createKnnIndex(INDEX_NAME, buildHalfFloatNestedMapping("flat", null));
        indexNestedTestDocs();

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildNestedSearchQuery(queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), NESTED_VECTOR_PATH);

        assertEquals(3, results.size());
        // Doc 1's vector is nearest to the query vector (all values closest to the origin).
        assertEquals("1", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testNestedSearch_whenHalfFloatHnsw_thenReturnsExpectedResults() {
        createKnnIndex(INDEX_NAME, buildHalfFloatNestedMapping("hnsw", "lucene"));
        indexNestedTestDocs();

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildNestedSearchQuery(queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), NESTED_VECTOR_PATH);

        assertEquals(3, results.size());
        assertEquals("1", results.get(0).getDocId());
    }

    private void indexNestedTestDocs() throws Exception {
        addKnnDocWithNestedField(INDEX_NAME, "1", NESTED_VECTOR_PATH, new Float[] { 1.0f, 1.0f, 1.0f, 1.0f });
        addKnnDocWithNestedField(INDEX_NAME, "2", NESTED_VECTOR_PATH, new Float[] { 2.0f, 2.0f, 2.0f, 2.0f });
        addKnnDocWithNestedField(INDEX_NAME, "3", NESTED_VECTOR_PATH, new Float[] { 5.0f, 5.0f, 5.0f, 5.0f });
    }

    private String buildHalfFloatNestedMapping(String methodName, String engine) throws Exception {
        KNNJsonIndexMappingsBuilder.Method.MethodBuilder methodBuilder = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(methodName)
            .spaceType("l2");
        if (engine != null) {
            methodBuilder.engine(engine);
        }
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(VECTOR_FIELD_NAME)
            .nestedFieldName(NESTED_FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType("half_float")
            .method(methodBuilder.build())
            .build()
            .getIndexMapping();
    }

    private XContentBuilder buildNestedSearchQuery(float[] queryVector, int k) throws Exception {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", NESTED_FIELD_NAME)
            .startObject("query")
            .startObject("knn")
            .startObject(NESTED_VECTOR_PATH)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
    }
}
