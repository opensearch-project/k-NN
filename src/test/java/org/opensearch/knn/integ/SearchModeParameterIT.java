/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.util.Arrays;

import static org.hamcrest.Matchers.containsString;
import static org.opensearch.knn.common.KNNConstants.SEARCH_MODE;
import static org.opensearch.knn.common.KNNConstants.EXACT_SEARCH_KEY;
import static org.opensearch.knn.common.KNNConstants.PROPERTIES;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;

public class SearchModeParameterIT extends KNNRestTestCase {
    private final static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f };
    private final static int DIMENSION = 2;
    private final static int K = 1;
    private static final String INDEX_NAME = "search-mode-index";
    private static final SpaceType TEST_SPACE_TYPE_L2 = SpaceType.L2;

    @SneakyThrows
    public void testCreateIndexWithSearchMode() {
        // exact search mode (testing both ann and exact search queries)
        createTestIndexWithSearchMode("exact");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        ResponseException ex = expectThrows(ResponseException.class, () -> validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K));
        String expMessage = "ANN search cannot be performed on exact search indices";
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString(expMessage));

        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);

        // ann search mode (testing both ann and exact search queries)
        createTestIndexWithSearchMode("ann");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testCreateIndexWithoutSearchMode() {
        createTestIndexWithNoSearchMode();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testNonKnnIndices() {
        createNonKnnIndex("exact");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteKNNIndex(INDEX_NAME);

        createNonKnnIndex("ann");
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteKNNIndex(INDEX_NAME);

        createNonKnnIndex(null);
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNScriptScoreSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K, TEST_SPACE_TYPE_L2);
        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testSearchMode_ExactAndLucene() throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject("my_vector1")
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, "hnsw")
            .field(KNN_ENGINE, "lucene")
            .endObject()
            .endObject()
            .startObject("my_vector2")
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(SEARCH_MODE, EXACT_SEARCH_KEY)
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(INDEX_NAME, builder.toString());

        Float[] vector1 = { 1.0f, 2.0f };
        Float[] vector2 = { 3.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "0", Arrays.asList("my_vector1", "my_vector2"), Arrays.asList(vector1, vector2));

        // lucene field
        validateKNNSearch(INDEX_NAME, "my_vector1", DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, "my_vector1", DIMENSION, 1, K, TEST_SPACE_TYPE_L2);

        // exact field
        ResponseException ex = expectThrows(ResponseException.class, () -> validateKNNSearch(INDEX_NAME, "my_vector2", DIMENSION, 1, K));
        String expMessage = "ANN search cannot be performed on exact search indices";
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString(expMessage));
        validateKNNScriptScoreSearch(INDEX_NAME, "my_vector2", DIMENSION, 1, K, TEST_SPACE_TYPE_L2);

        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testSearchMode_ExactAndFaiss() throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject("my_vector1")
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, "hnsw")
            .field(KNN_ENGINE, "faiss")
            .endObject()
            .endObject()
            .startObject("my_vector2")
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(SEARCH_MODE, EXACT_SEARCH_KEY)
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(INDEX_NAME, builder.toString());

        Float[] vector1 = { 1.0f, 2.0f };
        Float[] vector2 = { 3.0f, 6.0f };
        addKnnDoc(INDEX_NAME, "0", Arrays.asList("my_vector1", "my_vector2"), Arrays.asList(vector1, vector2));

        // lucene field
        validateKNNSearch(INDEX_NAME, "my_vector1", DIMENSION, 1, K);
        validateKNNScriptScoreSearch(INDEX_NAME, "my_vector1", DIMENSION, 1, K, TEST_SPACE_TYPE_L2);

        // exact field
        ResponseException ex = expectThrows(ResponseException.class, () -> validateKNNSearch(INDEX_NAME, "my_vector2", DIMENSION, 1, K));
        String expMessage = "ANN search cannot be performed on exact search indices";
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString(expMessage));
        validateKNNScriptScoreSearch(INDEX_NAME, "my_vector2", DIMENSION, 1, K, TEST_SPACE_TYPE_L2);

        deleteKNNIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testInvalidSearchModeIndexCreation() throws Exception {
        ResponseException ex = expectThrows(ResponseException.class, () -> createTestIndexWithSearchMode("exactsearch"));
        String expMessage = "Search mode must be either 'exact' or 'ann'";
        assertThat(EntityUtils.toString(ex.getResponse().getEntity()), containsString(expMessage));
    }

    private void createNonKnnIndex(String searchMode) throws IOException {
        Settings settings = Settings.builder().put(createKNNDefaultScriptScoreSettings()).build();

        if (searchMode == null) {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            createKnnIndex(INDEX_NAME, settings, mapping);
        } else {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .field(KNNConstants.SEARCH_MODE, searchMode)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            createKnnIndex(INDEX_NAME, settings, mapping);
        }
    }

    private void createTestIndexWithSearchMode(String searchMode) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(KNNConstants.SEARCH_MODE, searchMode)
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithNoSearchMode() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

}
