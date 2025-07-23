/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;

import static org.opensearch.knn.TestUtils.*;
import static org.opensearch.knn.common.KNNConstants.INDEX_PARAMETER_KEY;

public class IndexParameterIT extends KNNRestTestCase {
    private final static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f };
    private final static int DIMENSION = 2;
    private final static int K = 1;
    private static final String INDEX_NAME = "test_index";
    private static final String FIELD_NAME = "test_field";
    private static final SpaceType TEST_SPACE_TYPE_L2 = SpaceType.L2;

    // approximate_threshold = -1 tests
    @SneakyThrows
    public void testIndexFalse_ApproxThresholdNegOne_thenNoGraphsBuilt_NoException_Faiss() {
        createTestIndexWithIndexParameterWithEngine(false, -1, "faiss");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        // the following search does not throw an exception because of KNNWeight's isExactSearchRequire method:
        // since no native engine files are created, defaults to exact search
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testIndexTrue_ApproxThresholdNegOne_thenNoGraphsBuilt_NoException_Faiss() {
        createTestIndexWithIndexParameterWithEngine(true, -1, "faiss");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        // the following search does not throw an exception because of KNNWeight's isExactSearchRequire method:
        // since no native engine files are created, defaults to exact search
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testIndexFalse_ApproxThresholdNegOne_thenNoGraphsBuilt_NoException_Lucene() {
        createTestIndexWithIndexParameterWithEngine(false, -1, "lucene");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        // the following search does not throw an exception because of KNNWeight's isExactSearchRequire method:
        // since no native engine files are created, defaults to exact search
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testIndexTrue_ApproxThresholdNegOne_thenGraphsBuilt_NoException_Lucene() {
        createTestIndexWithIndexParameterWithEngine(true, -1, "lucene");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        // the following search does not throw an exception because since index = true, fall back to approx_threshold
        // and lucene doesn't pick up that setting so graphs are built
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    // approximate_threshold > 0 tests
    @SneakyThrows
    public void testIndexFalse_ApproxThresholdAboveZero_DocCountLess_thenNoGraphsBuilt_NoException_Faiss() {
        createTestIndexWithIndexParameterWithEngine(false, 2, "faiss");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        // the following search does not throw an exception because of KNNWeight's isExactSearchRequire method:
        // since no native engine files are created, defaults to exact search
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testIndexTrue_ApproxThresholdAboveZero_DocCountLess_thenNoGraphsBuilt_NoException_Faiss() {
        createTestIndexWithIndexParameterWithEngine(true, 2, "faiss");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        // the following search does not throw an exception because of KNNWeight's isExactSearchRequire method:
        // since no native engine files are created, defaults to exact search
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testIndexFalse_ApproxThresholdAboveZero_DocCountMore_thenNoGraphsBuilt_Faiss() {
        createTestIndexWithIndexParameterWithEngine(false, 1, "faiss");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testIndexFalse_ApproxThresholdAboveZero_DocCountLess_thenNoGraphsBuilt_Lucene() {
        createTestIndexWithIndexParameterWithEngine(false, 1, "lucene");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    // approximate_threshold = 0 tests
    @SneakyThrows
    public void testIndexFalse_ApproxThresholdZero_thenNoGraphsBuilt_Faiss() {
        createTestIndexWithIndexParameterWithEngine(false, 0, "faiss");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        // the following search does not throw an exception because of KNNWeight's isExactSearchRequire method:
        // since no native engine files are created, defaults to exact search
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testIndexFalse_ApproxThresholdZero_thenNoGraphsBuilt_Lucene() {
        createTestIndexWithIndexParameterWithEngine(false, 0, "lucene");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);

        // the following search does not throw an exception because of KNNWeight's isExactSearchRequire method:
        // since no native engine files are created, defaults to exact search
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testCreateIndexWithoutIndexParameter_Faiss() {
        createTestIndexWithNoIndexParameter("faiss");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testCreateIndexWithoutIndexParameter_Lucene() {
        createTestIndexWithNoIndexParameter("lucene");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testNonKnnIndices() {
        createNonKnnIndex(false);
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteKNNIndex(INDEX_NAME);

        createNonKnnIndex(true);
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteKNNIndex(INDEX_NAME);
    }

    private void createNonKnnIndex(boolean indexed) throws IOException {
        Settings settings = Settings.builder()
            .put(NUMBER_OF_SHARDS, 1)
            .put(NUMBER_OF_REPLICAS, 0)
            .put(INDEX_KNN, false)
            .put("index.use_compound_file", false)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(INDEX_PARAMETER_KEY, indexed)
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private void createTestIndexWithIndexParameterWithEngine(boolean indexed, int approxThreshold, String engine) throws IOException {
        Settings settings;
        Settings.Builder settingsBuilder = Settings.builder().put("index.use_compound_file", false);
        settingsBuilder.put(buildKNNIndexSettings(approxThreshold));
        settings = settingsBuilder.build();

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(INDEX_PARAMETER_KEY, indexed)
            .startObject("method")
            .field("engine", engine)
            .field("name", "hnsw")
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, settings, mapping);
    }

    private void createTestIndexWithNoIndexParameter(String engine) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field("engine", engine)
            .field("name", "hnsw")
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

}
