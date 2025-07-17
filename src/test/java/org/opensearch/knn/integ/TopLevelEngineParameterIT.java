/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;

public class TopLevelEngineParameterIT extends KNNRestTestCase {
    private final static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f };
    private final static int DIMENSION = 2;
    private final static int K = 1;
    private static final String INDEX_NAME = "top-level-engine-index";

    @SneakyThrows
    public void testBaseCase() {
        createTestIndexWithTopLevelEngineOnly();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        createTestIndexWithMethodEngineOnly();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        createTestIndexWithTopLevelEngineAndMethodEngine();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        createTestIndexWithNoEngine();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);
    }

    @SneakyThrows
    public void testDifferentEnginesDefined_ThenException() {
        Exception e = expectThrows(Exception.class, () -> createTestIndexWithDifferentEngines());
        assertTrue(
            e.getMessage(),
            e.getMessage().contains("Engine in method and top level engine should be same or one of them should be defined.")
        );
    }

    private void createTestIndexWithTopLevelEngineOnly() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithMethodEngineOnly() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithTopLevelEngineAndMethodEngine() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithNoEngine() throws IOException {
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

    private void createTestIndexWithDifferentEngines() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(TOP_LEVEL_PARAMETER_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }
}
