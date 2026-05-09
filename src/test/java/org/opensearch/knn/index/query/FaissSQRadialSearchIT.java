/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;

import java.util.concurrent.ThreadLocalRandom;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

/**
 * Integration tests verifying that radial search (max_distance / min_score) works on
 * Faiss 32x SQ quantized indices with HNSW method.
 *
 * These tests validate that the query pipeline executes without error and returns results.
 * Recall/precision validation will be added once ExactSearcher rescoring is implemented.
 */
public class FaissSQRadialSearchIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "faiss_sq_radial_search_test";
    private static final String FIELD_NAME = "vec_field";
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 100;

    // max_distance for L2 is squared Euclidean distance.
    // With dim=128, vectors in [-2,2], typical distance between random vectors is ~340.
    // Use 10000.0 to ensure all docs are within radius.
    private static final float LARGE_MAX_DISTANCE = 10000.0f;
    // min_score for L2 is 1/(1+distance). A very small value accepts nearly all docs.
    private static final float SMALL_MIN_SCORE = 0.0001f;

    @SneakyThrows
    public void testRadialSearch_withMaxDistance_onFaissSQHnsw() {
        createFaissSQHnswIndex();
        indexDocuments();
        refreshIndex(INDEX_NAME);

        Response response = executeRadialSearch("max_distance", LARGE_MAX_DISTANCE);
        assertEquals(200, response.getStatusLine().getStatusCode());
        assertTrue(getHitCount(response) > 0);

        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testRadialSearch_withMinScore_onFaissSQHnsw() {
        createFaissSQHnswIndex();
        indexDocuments();
        refreshIndex(INDEX_NAME);

        Response response = executeRadialSearch("min_score", SMALL_MIN_SCORE);
        assertEquals(200, response.getStatusLine().getStatusCode());
        assertTrue(getHitCount(response) > 0);

        deleteKNNIndex(INDEX_NAME);
    }

    private void createFaissSQHnswIndex() throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("data_type", "float")
            .field("mode", "on_disk")
            .field("compression_level", "32x")
            .field("space_type", SpaceType.L2.getValue())
            .startObject("method")
            .field("engine", "faiss")
            .field("name", "hnsw")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put(KNN_INDEX, true).build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private void indexDocuments() throws Exception {
        for (int i = 0; i < NUM_DOCS; i++) {
            float[] vector = new float[DIMENSION];
            for (int j = 0; j < DIMENSION; j++) {
                vector[j] = ThreadLocalRandom.current().nextFloat() * 4 - 2;
            }
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, vector);
        }
    }

    private Response executeRadialSearch(String thresholdType, float thresholdValue) throws Exception {
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

        Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
        request.setJsonEntity(query);
        return client().performRequest(request);
    }

    private int getHitCount(Response response) throws Exception {
        String responseBody = EntityUtils.toString(response.getEntity());
        return parseSearchResponse(responseBody, FIELD_NAME).size();
    }
}
