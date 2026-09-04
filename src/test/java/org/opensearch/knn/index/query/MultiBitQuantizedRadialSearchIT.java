/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.Randomness;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;

import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

/**
 * Radial search integration tests across the full quantized matrix served by the size-bounded rescoring
 * path: SQ and BQ at 1, 2 and 4 bits.
 *
 * <p>{@link FaissSQRadialSearchIT} and {@link LuceneSQRadialSearchIT} only cover 1-bit SQ expressed via
 * {@code mode}/{@code compression_level}. These cases set the encoder and {@code bits} explicitly, which is
 * the only way to reach 2-bit and 4-bit SQ.</p>
 */
public class MultiBitQuantizedRadialSearchIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "multibit_quantized_radial_search_test";
    private static final String FIELD_NAME = "vec_field";
    private static final int DIMENSION = 16;
    private static final int NUM_DOCS = 50;
    // Loose thresholds so the assertion is "radial search runs and returns in-radius hits", not a recall bar.
    private static final float LARGE_MAX_DISTANCE = 10000.0f;
    private static final float SMALL_MIN_SCORE = 0.0001f;

    private static final String[] ENCODERS = { "sq", "binary" };
    private static final int[] BITS = { 1, 2, 4 };

    @SneakyThrows
    public void testRadialSearch_withMaxDistance_acrossQuantizedMatrix() {
        forEachConfiguration((encoder, bits) -> {
            final Response response = executeRadialSearch("max_distance", LARGE_MAX_DISTANCE);
            assertEquals("max_distance on " + encoder + " " + bits + "-bit", 200, response.getStatusLine().getStatusCode());
            assertTrue("expected hits for " + encoder + " " + bits + "-bit", getHitCount(response) > 0);
        });
    }

    @SneakyThrows
    public void testRadialSearch_withMinScore_acrossQuantizedMatrix() {
        forEachConfiguration((encoder, bits) -> {
            final Response response = executeRadialSearch("min_score", SMALL_MIN_SCORE);
            assertEquals("min_score on " + encoder + " " + bits + "-bit", 200, response.getStatusLine().getStatusCode());
            assertTrue("expected hits for " + encoder + " " + bits + "-bit", getHitCount(response) > 0);
        });
    }

    /**
     * Radial search bounds its first pass at {@code size * oversample_factor}, so the returned hit count must
     * respect the requested size even though every document is inside the radius.
     */
    @SneakyThrows
    public void testRadialSearch_whenSizeIsSet_thenHonorsSize_acrossQuantizedMatrix() {
        final int size = 5;
        forEachConfiguration((encoder, bits) -> {
            final String query = XContentFactory.jsonBuilder()
                .startObject()
                .field("size", size)
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", randomVector())
                .field("max_distance", LARGE_MAX_DISTANCE)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();
            final Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
            request.setJsonEntity(query);
            final Response response = client().performRequest(request);

            assertEquals(200, response.getStatusLine().getStatusCode());
            final int hits = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME).size();
            assertTrue("hits (" + hits + ") must not exceed size for " + encoder + " " + bits + "-bit", hits <= size);
            assertTrue("expected hits for " + encoder + " " + bits + "-bit", hits > 0);
        });
    }

    /**
     * {@code oversample_factor} sizes the first pass for radial search, so it must be accepted alongside a
     * radial threshold rather than rejected as mutually exclusive.
     */
    @SneakyThrows
    public void testRadialSearch_withOversampleFactor_acrossQuantizedMatrix() {
        forEachConfiguration((encoder, bits) -> {
            final String query = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", randomVector())
                .field("max_distance", LARGE_MAX_DISTANCE)
                .startObject("rescore")
                .field("oversample_factor", 3.0f)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();
            final Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
            request.setJsonEntity(query);
            final Response response = client().performRequest(request);

            assertEquals("oversample_factor on " + encoder + " " + bits + "-bit", 200, response.getStatusLine().getStatusCode());
            assertTrue("expected hits for " + encoder + " " + bits + "-bit", getHitCount(response) > 0);
        });
    }

    @FunctionalInterface
    private interface ConfigurationAssertion {
        void run(String encoder, int bits) throws Exception;
    }

    /** Builds, populates and tears down an index for each {encoder, bits} pair, running the assertion on each. */
    private void forEachConfiguration(final ConfigurationAssertion assertion) throws Exception {
        for (final String encoder : ENCODERS) {
            for (final int bits : BITS) {
                createQuantizedIndex(encoder, bits);
                try {
                    indexDocuments();
                    refreshIndex(INDEX_NAME);
                    assertion.run(encoder, bits);
                } finally {
                    deleteKNNIndex(INDEX_NAME);
                }
            }
        }
    }

    private void createQuantizedIndex(final String encoder, final int bits) throws Exception {
        final String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("data_type", "float")
            .field("space_type", SpaceType.L2.getValue())
            .startObject("method")
            .field("engine", "faiss")
            .field("name", "hnsw")
            .startObject("parameters")
            .startObject("encoder")
            .field("name", encoder)
            .startObject("parameters")
            .field("bits", bits)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        final Settings settings = Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put(KNN_INDEX, true).build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private void indexDocuments() throws Exception {
        for (int i = 0; i < NUM_DOCS; i++) {
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, randomVector());
        }
    }

    private Float[] randomVector() {
        final Float[] vector = new Float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = Randomness.get().nextFloat() * 4 - 2;
        }
        return vector;
    }

    private Response executeRadialSearch(final String thresholdType, final float thresholdValue) throws Exception {
        final String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", randomVector())
            .field(thresholdType, thresholdValue)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        final Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
        request.setJsonEntity(query);
        return client().performRequest(request);
    }

    private int getHitCount(final Response response) throws Exception {
        return parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME).size();
    }
}
