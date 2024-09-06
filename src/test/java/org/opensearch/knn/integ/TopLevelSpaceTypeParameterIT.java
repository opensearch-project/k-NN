/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;

public class TopLevelSpaceTypeParameterIT extends KNNRestTestCase {
    private final static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f };
    private final static int DIMENSION = 2;
    private final static int K = 1;
    private static final String INDEX_NAME = "top-level-space-type-index";

    @SneakyThrows
    public void testBaseCase() {
        createTestIndexWithTopLevelSpaceTypeOnly();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        createTestIndexWithTopLevelSpaceTypeAndMethodSpaceType();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);

        createTestIndexWithNoSpaceType();
        addKnnDoc(INDEX_NAME, "0", FIELD_NAME, TEST_VECTOR);
        validateKNNSearch(INDEX_NAME, FIELD_NAME, DIMENSION, 1, K);
        deleteIndex(INDEX_NAME);
    }

    private void createTestIndexWithTopLevelSpaceTypeOnly() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithTopLevelSpaceTypeAndMethodSpaceType() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }

    private void createTestIndexWithNoSpaceType() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }
}
