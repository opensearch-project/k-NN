/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;

public class ModeAndCompressionIT extends KNNRestTestCase {

    private static final int DIMENSION = 10;

    public void testIndexCreation() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "16x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME + "1", mapping);

        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "in_memory")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        mapping = builder.toString();
        createKnnIndex(INDEX_NAME + "2", mapping);

        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "invalid")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .startObject(PARAMETERS)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        String finalMapping = builder.toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME + "3", finalMapping));
    }

    @SneakyThrows
    public void testTraining() {
        String trainingIndexName = "training-index";
        String trainingFieldName = "training-field";
        String modelDescription = "test model";
        int dimension = 20;
        int trainingDataCount = 256;
        createBasicKnnIndex(trainingIndexName, trainingFieldName, dimension);
        bulkIngestRandomVectors(trainingIndexName, trainingFieldName, trainingDataCount, dimension);

        String modelId1 = "test-model-1";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, trainingIndexName)
            .field(TRAIN_FIELD_PARAMETER, trainingFieldName)
            .field(KNNConstants.DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .endObject()
            .field(MODEL_DESCRIPTION, modelDescription)
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .endObject();
        Response trainResponse = trainModel(modelId1, xContentBuilder);
        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
        assertTrainingSucceeds(modelId1, 360, 1000);
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("model_id", modelId1)
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        createKnnIndex(INDEX_NAME + "1", mapping);
        deleteKNNIndex(INDEX_NAME + "1");
        deleteModel(modelId1);
        String modelId2 = "test-model-2";
        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, trainingIndexName)
            .field(TRAIN_FIELD_PARAMETER, trainingFieldName)
            .field(KNNConstants.DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .endObject()
            .field(MODEL_DESCRIPTION, modelDescription)
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .field(MODE_PARAMETER, "invalid")
            .endObject();
        expectThrows(ResponseException.class, () -> trainModel(modelId2, xContentBuilder2));
    }
}
