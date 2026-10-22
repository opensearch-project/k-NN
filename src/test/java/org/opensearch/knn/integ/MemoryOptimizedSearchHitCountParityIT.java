/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * Toggling {@code index.knn.memory_optimized_search} must not change which documents a {@code knn} clause matches,
 * so {@code hits.total} and an aggregation over the clause must be the same with and without it. Two indices with
 * identical data and segment layout are compared, one per setting value.
 *
 * <p>Without memory-optimized search, a query with a rescore context or nested expansion is reduced to the shard's
 * global top k, and any other query returns each segment's own top k. Each scenario below exercises one of those.
 *
 * <p>This cluster sets {@code mem_opt_srch.force_reenter}, which re-searches every segment with the full k in the
 * optimistic second pass, so these tests do not cover segments that keep only their pro-rated first-pass budget.
 */
public class MemoryOptimizedSearchHitCountParityIT extends KNNRestTestCase {
    private static final String NESTED_PATH = "nested_field";
    private static final String NESTED_VECTOR_FIELD = NESTED_PATH + ".vec";
    private static final String COUNTER_FIELD = "counter";
    private static final String BUCKET_FIELD = "bucket";
    private static final String AGGREGATION_NAME = "matched_docs";
    private static final int DIMENSION = 8;
    private static final int K = 100;
    private static final int NUM_SEGMENTS = 4;
    private static final int DOCS_PER_SEGMENT = 2 * K;

    @SneakyThrows
    public void testFlatField_hitCountMatchesNonMemoryOptimizedSearch() {
        final String off = "mos-parity-flat-off";
        final String on = "mos-parity-flat-on";
        final float[] query = randomVector(new Random(7L));
        for (String index : new String[] { off, on }) {
            createParityIndex(index, index.equals(on), false);
            ingest(index, false);
        }
        assertTrue("the shard must hold more than one segment for this test to mean anything", getTotalSegmentCount(off) > 1);
        assertEquals("both indices must have the same segment layout", getTotalSegmentCount(off), getTotalSegmentCount(on));

        assertSameMatches("plain query, k >= ef_search", off, on, knnClause(FIELD_NAME, query, K, ""), null);
        assertSameMatches("plain query, k < ef_search", off, on, knnClause(FIELD_NAME, query, K / 2, ""), null);
        assertSameMatches(
            "filtered query",
            off,
            on,
            knnClause(FIELD_NAME, query, K, ",\"filter\":{\"terms\":{\"" + BUCKET_FIELD + "\":[0,1]}}"),
            null
        );
        assertSameMatches("rescored query", off, on, knnClause(FIELD_NAME, query, K, ",\"rescore\":{\"oversample_factor\":2.0}"), null);

        forceMergeKnnIndex(off, 1);
        forceMergeKnnIndex(on, 1);
        assertSameMatches("plain query after force merge", off, on, knnClause(FIELD_NAME, query, K, ""), null);

        deleteKNNIndex(off);
        deleteKNNIndex(on);
    }

    @SneakyThrows
    public void testNestedField_hitCountMatchesNonMemoryOptimizedSearch() {
        final String off = "mos-parity-nested-off";
        final String on = "mos-parity-nested-on";
        final float[] query = randomVector(new Random(7L));
        for (String index : new String[] { off, on }) {
            createParityIndex(index, index.equals(on), true);
            ingest(index, true);
        }
        assertTrue("the shard must hold more than one segment for this test to mean anything", getTotalSegmentCount(off) > 1);
        assertEquals("both indices must have the same segment layout", getTotalSegmentCount(off), getTotalSegmentCount(on));

        assertSameMatches("nested query", off, on, knnClause(NESTED_VECTOR_FIELD, query, K, ""), NESTED_PATH);
        assertSameMatches(
            "nested query with expand_nested_docs",
            off,
            on,
            knnClause(NESTED_VECTOR_FIELD, query, K, ",\"expand_nested_docs\":true"),
            NESTED_PATH
        );

        deleteKNNIndex(off);
        deleteKNNIndex(on);
    }

    private void assertSameMatches(String scenario, String off, String on, String knnClause, String nestedPath) throws IOException {
        final long[] expected = search(off, knnClause, nestedPath);
        final long[] actual = search(on, knnClause, nestedPath);
        assertEquals(scenario + ": hits.total must not change with memory-optimized search", expected[0], actual[0]);
        assertEquals(scenario + ": the aggregation must not change with memory-optimized search", expected[1], actual[1]);
    }

    private static String knnClause(String field, float[] vector, int k, String extra) {
        return "{\"knn\":{\"" + field + "\":{\"vector\":" + Arrays.toString(vector) + ",\"k\":" + k + extra + "}}}";
    }

    @SneakyThrows
    private long[] search(String index, String knnClause, String nestedPath) {
        final String query = nestedPath == null ? knnClause : "{\"nested\":{\"path\":\"" + nestedPath + "\",\"query\":" + knnClause + "}}";
        final String body = "{\"size\":10,\"query\":"
            + query
            + ",\"aggs\":{\""
            + AGGREGATION_NAME
            + "\":{\"value_count\":{\"field\":\""
            + COUNTER_FIELD
            + "\"}}}}";
        final Response response = performSearch(index, body);
        final String responseBody = EntityUtils.toString(response.getEntity());
        @SuppressWarnings("unchecked")
        final Map<String, Object> hits = (Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody
        ).map().get("hits");
        @SuppressWarnings("unchecked")
        final long total = ((Number) ((Map<String, Object>) hits.get("total")).get("value")).longValue();
        return new long[] { total, parseAggregationResponse(responseBody, AGGREGATION_NAME).longValue() };
    }

    @SneakyThrows
    private void createParityIndex(String index, boolean memoryOptimizedSearch, boolean nested) {
        final Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.knn.advanced.approximate_threshold", 0)
            .put("index.knn.memory_optimized_search", memoryOptimizedSearch)
            // Keep both indices at the same segment layout: no background merges below 50 segments.
            .put("index.merge.policy.segments_per_tier", 50)
            .build();
        final XContentBuilder mapping = XContentFactory.jsonBuilder().startObject().startObject("properties");
        if (nested) {
            mapping.startObject(NESTED_PATH).field("type", "nested").startObject("properties").startObject("vec");
            vectorField(mapping).endObject().endObject().endObject();
        } else {
            mapping.startObject(FIELD_NAME);
            vectorField(mapping).endObject();
        }
        mapping.startObject(COUNTER_FIELD).field("type", "integer").endObject();
        mapping.startObject(BUCKET_FIELD).field("type", "integer").endObject();
        mapping.endObject().endObject();
        createKnnIndex(index, settings, mapping.toString());
    }

    private static XContentBuilder vectorField(XContentBuilder mapping) throws IOException {
        return mapping.field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("name", METHOD_HNSW)
            .field("engine", "faiss")
            .field("space_type", "l2")
            .endObject();
    }

    @SneakyThrows
    private void ingest(String index, boolean nested) {
        final Random random = new Random(42L);
        for (int segment = 0; segment < NUM_SEGMENTS; ++segment) {
            final StringBuilder body = new StringBuilder();
            for (int i = 0; i < DOCS_PER_SEGMENT; ++i) {
                final int docId = segment * DOCS_PER_SEGMENT + i;
                final XContentBuilder doc = XContentFactory.jsonBuilder()
                    .startObject()
                    .field(COUNTER_FIELD, docId)
                    .field(BUCKET_FIELD, docId % 4);
                if (nested) {
                    doc.startArray(NESTED_PATH);
                    for (int child = 0; child < 2; ++child) {
                        doc.startObject().field("vec", randomVector(random)).endObject();
                    }
                    doc.endArray();
                } else {
                    doc.field(FIELD_NAME, randomVector(random));
                }
                doc.endObject();
                body.append("{\"index\":{\"_id\":\"").append(docId).append("\"}}\n").append(doc.toString()).append("\n");
            }
            final Request request = new Request("POST", "/" + index + "/_bulk");
            request.addParameter("refresh", "true");
            request.setJsonEntity(body.toString());
            assertEquals(RestStatus.OK, RestStatus.fromCode(client().performRequest(request).getStatusLine().getStatusCode()));
            flushIndex(index);
        }
        refreshIndex(index);
    }

    private static float[] randomVector(final Random random) {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; ++i) {
            vector[i] = random.nextFloat();
        }
        return vector;
    }
}
