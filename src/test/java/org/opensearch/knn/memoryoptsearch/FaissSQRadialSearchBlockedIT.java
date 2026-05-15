/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.opensearch.client.Request;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;

import java.util.concurrent.ThreadLocalRandom;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

/**
 * Integration test to verify that radial search (max_distance / min_score) is blocked
 * for Faiss SQ (scalar quantized, 32x compression) indices.
 */
public class FaissSQRadialSearchBlockedIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "faiss_sq_radial_test";
    private static final String FIELD_NAME = "vec_field";
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 100;

    @SneakyThrows
    public void testRadialSearch_withMaxDistance_onFaissSQ32x_thenBlocked() {
        // Create Faiss SQ index
        createFaissSQIndex();

        // Index vectors + refresh
        indexDocuments();
        refreshIndex(INDEX_NAME);

        // Radial search with max_distance should fail
        final float[] queryVector = generateRandomVector();
        final String query = buildRadialSearchQuery(FIELD_NAME, queryVector, "max_distance", 100.0f);
        final Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
        request.setJsonEntity(query);

        // Exception should be thrown, radial search is blocked on Faiss SQ
        final ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(ex.getMessage().contains("Radial search is not supported for indices which have quantization enabled"));

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testRadialSearch_withMinScore_onFaissSQ32x_thenBlocked() {
        // Create Faiss SQ index
        createFaissSQIndex();

        // Index vectors + refresh
        indexDocuments();
        refreshIndex(INDEX_NAME);

        // Radial search with min_score should fail
        float[] queryVector = generateRandomVector();
        final String query = buildRadialSearchQuery(FIELD_NAME, queryVector, "min_score", 0.01f);
        final Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
        request.setJsonEntity(query);

        // Exception should be thrown, radial search is blocked on Faiss SQ
        final ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(ex.getMessage().contains("Radial search is not supported for indices which have quantization enabled"));

        // Delete index
        deleteKNNIndex(INDEX_NAME);
    }

    private void createFaissSQIndex() throws Exception {
        // Faiss SQ is default for 32x
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

        // Index setting
        Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put(KNN_INDEX, true).build();

        // Create index
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private void indexDocuments() throws Exception {
        for (int i = 0; i < NUM_DOCS; i++) {
            final float[] vector = new float[DIMENSION];
            for (int j = 0; j < DIMENSION; j++) {
                vector[j] = ThreadLocalRandom.current().nextFloat() * 4 - 2;
            }
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, vector);
        }
    }

    private float[] generateRandomVector() {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = ThreadLocalRandom.current().nextFloat() * 4 - 2;
        }
        return vector;
    }

    private String buildRadialSearchQuery(
        final String fieldName,
        final float[] vector,
        final String thresholdType,
        final float thresholdValue
    ) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder.startObject("query");
        builder.startObject("knn");
        builder.startObject(fieldName);
        builder.field("vector", vector);
        builder.field(thresholdType, thresholdValue);
        builder.endObject();
        builder.endObject();
        builder.endObject();
        builder.endObject();
        return builder.toString();
    }
}
