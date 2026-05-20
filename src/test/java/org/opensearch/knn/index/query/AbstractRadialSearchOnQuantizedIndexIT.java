/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Abstract base for radial search integration tests on quantized indices.
 * Subclasses provide the index creation logic; the test methods are shared.
 */
public abstract class AbstractRadialSearchOnQuantizedIndexIT extends KNNRestTestCase {

    protected static final String FIELD_NAME = "vec_field";
    protected static final int DIMENSION = 128;
    protected static final int NUM_DOCS = 100;
    protected static final float LARGE_MAX_DISTANCE = 10000.0f;
    protected static final float SMALL_MIN_SCORE = 0.0001f;

    protected abstract String getIndexName();

    /**
     * Creates the quantized index with default settings (no custom max_result_window).
     */
    protected abstract void createQuantizedIndex() throws Exception;

    /**
     * Creates the quantized index with a custom max_result_window setting.
     */
    protected abstract void createQuantizedIndexWithMaxResultWindow(int maxResultWindow) throws Exception;

    @SneakyThrows
    public void testRadialSearch_withMaxDistance() {
        createQuantizedIndex();
        indexDocuments();
        refreshIndex(getIndexName());

        Response response = executeRadialSearch("max_distance", LARGE_MAX_DISTANCE);
        assertEquals(200, response.getStatusLine().getStatusCode());
        assertTrue(getHitCount(response) > 0);

        deleteKNNIndex(getIndexName());
    }

    @SneakyThrows
    public void testRadialSearch_withMinScore() {
        createQuantizedIndex();
        indexDocuments();
        refreshIndex(getIndexName());

        Response response = executeRadialSearch("min_score", SMALL_MIN_SCORE);
        assertEquals(200, response.getStatusLine().getStatusCode());
        assertTrue(getHitCount(response) > 0);

        deleteKNNIndex(getIndexName());
    }

    // Given: index with max_result_window=100 and 200 docs all within radius
    // When: radial search is executed with size=100
    // Then: hits.hits size is <= 100 (honoring the index setting)
    @SneakyThrows
    public void testRadialSearch_whenMaxResultWindowIs100_thenResultsCappedAt100() {
        int maxResultWindow = 100;
        int numDocs = 200;

        createQuantizedIndexWithMaxResultWindow(maxResultWindow);

        // Index 200 docs — all identical vectors so all are within any radius
        float[] identicalVector = new float[DIMENSION];
        for (int j = 0; j < DIMENSION; j++) {
            identicalVector[j] = 1.0f;
        }
        for (int i = 0; i < numDocs; i++) {
            addKnnDoc(getIndexName(), Integer.toString(i), FIELD_NAME, identicalVector);
        }
        refreshIndex(getIndexName());

        // When: radial search with very large distance (all docs within radius)
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .field("size", maxResultWindow)
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", identicalVector)
            .field("max_distance", LARGE_MAX_DISTANCE)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Request request = new Request("POST", "/" + getIndexName() + "/_search");
        request.setJsonEntity(query);
        Response response = client().performRequest(request);

        // Then: parse hits.hits — results should be capped at max_result_window (100) by the rescore layer
        assertEquals(200, response.getStatusLine().getStatusCode());
        String responseBody = EntityUtils.toString(response.getEntity());
        int resultSize = parseSearchResponse(responseBody, FIELD_NAME).size();
        assertTrue("Results size should be <= max_result_window (100), got " + resultSize, resultSize <= maxResultWindow);

        deleteKNNIndex(getIndexName());
    }

    protected void indexDocuments() throws Exception {
        for (int i = 0; i < NUM_DOCS; i++) {
            float[] vector = new float[DIMENSION];
            for (int j = 0; j < DIMENSION; j++) {
                vector[j] = ThreadLocalRandom.current().nextFloat() * 4 - 2;
            }
            addKnnDoc(getIndexName(), Integer.toString(i), FIELD_NAME, vector);
        }
    }

    protected Response executeRadialSearch(String thresholdType, float thresholdValue) throws Exception {
        float[] queryVector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            queryVector[i] = ThreadLocalRandom.current().nextFloat() * 4 - 2;
        }

        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field(thresholdType, thresholdValue)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Request request = new Request("POST", "/" + getIndexName() + "/_search");
        request.setJsonEntity(query);
        return client().performRequest(request);
    }

    protected int getHitCount(Response response) throws Exception {
        String responseBody = EntityUtils.toString(response.getEntity());
        return parseSearchResponse(responseBody, FIELD_NAME).size();
    }
}
