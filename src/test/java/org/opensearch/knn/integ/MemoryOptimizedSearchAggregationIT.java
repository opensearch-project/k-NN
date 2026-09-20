/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.util.Map;
import java.util.Random;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;

/**
 * A {@code knn} clause is a filter on the documents an aggregation sees, so an aggregation over one must
 * observe exactly the top k docs of the shard - the same set the non-memory-optimized path produces, and
 * the same set before and after a merge.
 *
 * <p>{@code mode: on_disk} with {@code compression_level: 1x} forces memory-optimized search regardless of
 * the {@code index.knn.memory_optimized_search} index setting, and {@code k} is chosen to be not greater
 * than the default {@code ef_search} of 100, which is the combination that used to return the union of
 * every segment's hits.
 */
public class MemoryOptimizedSearchAggregationIT extends KNNRestTestCase {
    private static final String INDEX = "mos-aggregation-it";
    private static final String COUNTER_FIELD = "counter";
    private static final String AGGREGATION_NAME = "matched_docs";
    private static final int DIMENSION = 8;
    private static final int K = 100;
    private static final int NUM_SEGMENTS = 4;
    private static final int DOCS_PER_SEGMENT = 2 * K;

    @SneakyThrows
    public void testAggregationOverKnnClause_seesTopKOnly_andIsStableAcrossForceMerge() {
        createOnDiskIndex();

        final Random random = new Random(1234567L);
        for (int segment = 0; segment < NUM_SEGMENTS; ++segment) {
            bulkIndexSegment(segment, random);
            flushIndex(INDEX);
        }
        refreshIndex(INDEX);

        final int segmentCount = getTotalSegmentCount(INDEX);
        assertTrue("the shard must hold more than one segment for this test to mean anything", segmentCount > 1);

        final float[] queryVector = randomVector(random);

        // Every segment holds more than k docs, so an untrimmed union would hold k * segmentCount docs.
        assertMatchedDocCount("with " + segmentCount + " segments", queryVector);

        forceMergeKnnIndex(INDEX, 1);
        assertEquals("force merge should have left a single segment", 1, getTotalSegmentCount(INDEX));

        // The property a user notices breaking: routine maintenance must not change the answer.
        assertMatchedDocCount("after force merge to a single segment", queryVector);

        deleteKNNIndex(INDEX);
    }

    private void assertMatchedDocCount(final String context, final float[] queryVector) throws IOException {
        final String responseBody = searchWithAggregation(queryVector);
        assertEquals(
            "aggregation over the knn clause should see exactly k docs " + context,
            (double) K,
            parseAggregationResponse(responseBody, AGGREGATION_NAME),
            0.0
        );
        assertEquals("hits.total should be k " + context, K, parseTotalHits(responseBody));
    }

    @SneakyThrows
    private String searchWithAggregation(final float[] queryVector) {
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field("size", 10)
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field("k", K)
            .endObject()
            .endObject()
            .endObject()
            .startObject("aggs")
            .startObject(AGGREGATION_NAME)
            .startObject("value_count")
            .field("field", COUNTER_FIELD)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        final Response response = performSearch(INDEX, builder.toString());
        return EntityUtils.toString(response.getEntity());
    }

    @SuppressWarnings("unchecked")
    private long parseTotalHits(final String responseBody) throws IOException {
        final Map<String, Object> hits = (Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody
        ).map().get("hits");
        return ((Number) ((Map<String, Object>) hits.get("total")).get("value")).longValue();
    }

    @SneakyThrows
    private void createOnDiskIndex() {
        final String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x1.getName())
            .startObject("method")
            .field("name", METHOD_HNSW)
            .endObject()
            .endObject()
            .startObject(COUNTER_FIELD)
            .field("type", "integer")
            .endObject()
            .endObject()
            .endObject()
            .toString();

        createKnnIndex(INDEX, getDefaultIndexSettings(), mapping);
    }

    @SneakyThrows
    private void bulkIndexSegment(final int segment, final Random random) {
        final StringBuilder body = new StringBuilder();
        for (int i = 0; i < DOCS_PER_SEGMENT; ++i) {
            final int docId = segment * DOCS_PER_SEGMENT + i;
            final XContentBuilder doc = XContentFactory.jsonBuilder()
                .startObject()
                .field(FIELD_NAME, randomVector(random))
                .field(COUNTER_FIELD, docId)
                .endObject();
            body.append("{\"index\":{\"_id\":\"").append(docId).append("\"}}\n").append(doc.toString()).append("\n");
        }

        final Request request = new Request("POST", "/" + INDEX + "/_bulk");
        request.addParameter("refresh", "true");
        request.setJsonEntity(body.toString());
        final Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    private static float[] randomVector(final Random random) {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; ++i) {
            vector[i] = random.nextFloat();
        }
        return vector;
    }
}
