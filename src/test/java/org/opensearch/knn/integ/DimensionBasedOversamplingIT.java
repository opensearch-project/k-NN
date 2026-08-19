/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;
import java.util.Arrays;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

/**
 * Integration test verifying dimension-based oversampling applies correctly for disk-based
 * vector indices with 16x compression. Uses 16x because 32x resolves to SQ(bits=1) on 3.6+
 * which explicitly disables dimension-based override via allowOverrideOversampleFactor=false.
 */
public class DimensionBasedOversamplingIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "dimension-oversampling-test";
    private static final String FIELD_NAME = "test_vector";
    private static final int NUM_DOCS = 500;

    @SneakyThrows
    public void testOversamplingFactor_whenDimAbove1000_thenFirstPassKEqualsMinPassResults() {
        int dimension = 1024;
        try {
            createDiskBased16xIndex(dimension);
            bulkIngestRandomVectors(INDEX_NAME, FIELD_NAME, NUM_DOCS, dimension);
            refreshIndex(INDEX_NAME);

            String explanation = searchWithExplain(dimension, 10);
            assertFirstPassK(explanation, RescoreContext.MIN_FIRST_PASS_RESULTS);

            updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED, true));
            String explanationDisabled = searchWithExplain(dimension, 10);
            assertFirstPassK(explanationDisabled, RescoreContext.MIN_FIRST_PASS_RESULTS);
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    @SneakyThrows
    public void testOversamplingFactor_whenDim768to999_thenFirstPassKUses2xFactor() {
        int dimension = 800;
        try {
            createDiskBased16xIndex(dimension);
            bulkIngestRandomVectors(INDEX_NAME, FIELD_NAME, NUM_DOCS, dimension);
            refreshIndex(INDEX_NAME);

            String explanation = searchWithExplain(dimension, 100);
            assertFirstPassK(explanation, 200);

            updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED, true));
            String explanationDisabled = searchWithExplain(dimension, 100);
            assertFirstPassK(explanationDisabled, 200);
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    @SneakyThrows
    public void testOversamplingFactor_whenDimBelow768_thenFirstPassKUses3xFactor() {
        int dimension = 128;
        try {
            createDiskBased16xIndex(dimension);
            bulkIngestRandomVectors(INDEX_NAME, FIELD_NAME, NUM_DOCS, dimension);
            refreshIndex(INDEX_NAME);

            String explanation = searchWithExplain(dimension, 100);
            assertFirstPassK(explanation, 300);

            updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED, true));
            String explanationDisabled = searchWithExplain(dimension, 100);
            assertFirstPassK(explanationDisabled, 300);
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    @SneakyThrows
    public void testOversamplingFactor_whenUserProvided_thenNotOverridden() {
        int dimension = 1024;
        float userOversample = 5.0f;
        try {
            createDiskBased16xIndex(dimension);
            bulkIngestRandomVectors(INDEX_NAME, FIELD_NAME, NUM_DOCS, dimension);
            refreshIndex(INDEX_NAME);

            String explanation = searchWithExplainAndOversample(dimension, 100, userOversample);
            assertFirstPassK(explanation, 500);
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    @SneakyThrows
    public void testOversamplingFactor_32xBQ_whenDimAbove1000_thenFirstPassKEqualsMinPassResults() {
        int dimension = 1024;
        try {
            createDiskBased32xBQIndex(dimension);
            bulkIngestRandomVectors(INDEX_NAME, FIELD_NAME, NUM_DOCS, dimension);
            refreshIndex(INDEX_NAME);

            String explanation = searchWithExplain(dimension, 10);
            assertFirstPassK(explanation, RescoreContext.MIN_FIRST_PASS_RESULTS);

            updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED, true));
            String explanationDisabled = searchWithExplain(dimension, 10);
            assertFirstPassK(explanationDisabled, RescoreContext.MIN_FIRST_PASS_RESULTS);
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    @SneakyThrows
    public void testOversamplingFactor_32xBQ_whenDim768to999_thenFirstPassKUses2xFactor() {
        int dimension = 800;
        try {
            createDiskBased32xBQIndex(dimension);
            bulkIngestRandomVectors(INDEX_NAME, FIELD_NAME, NUM_DOCS, dimension);
            refreshIndex(INDEX_NAME);

            String explanation = searchWithExplain(dimension, 100);
            assertFirstPassK(explanation, 200);

            updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED, true));
            String explanationDisabled = searchWithExplain(dimension, 100);
            assertFirstPassK(explanationDisabled, 200);
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    @SneakyThrows
    public void testOversamplingFactor_32xBQ_whenDimBelow768_thenFirstPassKUses3xFactor() {
        int dimension = 128;
        try {
            createDiskBased32xBQIndex(dimension);
            bulkIngestRandomVectors(INDEX_NAME, FIELD_NAME, NUM_DOCS, dimension);
            refreshIndex(INDEX_NAME);

            String explanation = searchWithExplain(dimension, 100);
            assertFirstPassK(explanation, 300);

            updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED, true));
            String explanationDisabled = searchWithExplain(dimension, 100);
            assertFirstPassK(explanationDisabled, 300);
        } finally {
            deleteKNNIndex(INDEX_NAME);
        }
    }

    private void createDiskBased16xIndex(int dimension) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "16x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        Settings settings = Settings.builder().put(KNN_INDEX, true).build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private void createDiskBased32xBQIndex(int dimension) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, "binary")
            .startObject(PARAMETERS)
            .field("bits", 1)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        Settings settings = Settings.builder().put(KNN_INDEX, true).build();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private String searchWithExplain(int dimension, int k) throws IOException, ParseException {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, 1.0f);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response response = performSearch(INDEX_NAME, queryBuilder.toString(), "explain=true");
        return EntityUtils.toString(response.getEntity());
    }

    private String searchWithExplainAndOversample(int dimension, int k, float oversample) throws IOException, ParseException {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, 1.0f);

        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", queryVector)
            .field("k", k)
            .startObject("rescore")
            .field("oversample_factor", oversample)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response response = performSearch(INDEX_NAME, queryBuilder.toString(), "explain=true");
        return EntityUtils.toString(response.getEntity());
    }

    private void assertFirstPassK(String responseBody, int expectedFirstPassK) {
        String expectedString = "the first pass k was " + expectedFirstPassK;
        assertTrue("Expected explain to contain '" + expectedString + "' but got: " + responseBody, responseBody.contains(expectedString));
    }
}
