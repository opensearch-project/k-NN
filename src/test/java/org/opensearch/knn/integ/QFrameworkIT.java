/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class QFrameworkIT extends KNNRestTestCase {

    private final static float[] TEST_VECTOR = new float[] { 1.0f, 2.0f };
    private final static int DIMENSION = 2;
    private final static int K = 1;

    public void testBaseCase() throws IOException {
        createTestIndex(4);
        // TODO :- UnComment this once Search is Integrated and KNN_USE_LUCENE_VECTOR_FORMAT_ENABLED_SETTING is enabled
        // addKnnDoc(INDEX_NAME, "1", FIELD_NAME, TEST_VECTOR);
        // Response response = searchKNNIndex(
        // // INDEX_NAME,
        // // XContentFactory.jsonBuilder()
        // // .startObject()
        // // .startObject("query")
        // // .startObject("knn")
        // // .startObject(FIELD_NAME)
        // // .field("vector", TEST_VECTOR)
        // // .field("k", K)
        // // .endObject()
        // // .endObject()
        // // .endObject()
        // // .endObject(),
        // // 1
        // // );
        // assertOK(response);
    }

    private void createTestIndex(int bitCount) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, QFrameBitEncoder.NAME)
            .startObject(PARAMETERS)
            .field(QFrameBitEncoder.BITCOUNT_PARAM, bitCount)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME, mapping);
    }
}
