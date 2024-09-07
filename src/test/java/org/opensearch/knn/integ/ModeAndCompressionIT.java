/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Ignore;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.mapper.ModeBasedResolver;
import org.opensearch.knn.index.query.parser.RescoreParser;

import java.util.List;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class ModeAndCompressionIT extends KNNRestTestCase {

    private static final String TRAINING_INDEX_NAME = "training_index";
    private static final String TRAINING_FIELD_NAME = "training_field";
    private static final int TRAINING_VECS = 20;

    private static final int DIMENSION = 16;
    private static final int NUM_DOCS = 20;
    private static final int K = 2;
    private final static float[] TEST_VECTOR = new float[] {
        1.0f,
        2.0f,
        1.0f,
        2.0f,
        1.0f,
        2.0f,
        1.0f,
        2.0f,
        1.0f,
        2.0f,
        1.0f,
        2.0f,
        1.0f,
        2.0f,
        1.0f,
        2.0f };

    @SneakyThrows
    public void testIndexCreation_whenInvalid_thenFail() {
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
        String mapping1 = builder.toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping1));

        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(VECTOR_DATA_TYPE_FIELD, "byte")
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "16x")
            .endObject()
            .endObject()
            .endObject();
        String mapping2 = builder.toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping2));

        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "8x")
            .endObject()
            .endObject()
            .endObject();
        String mapping3 = builder.toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping3));

        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "on_disk1222")
            .endObject()
            .endObject()
            .endObject();
        String mapping4 = builder.toString();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, mapping4));
    }

    @SneakyThrows
    public void testIndexCreation_whenValid_ThenSucceed() {
        XContentBuilder builder;
        for (CompressionLevel compressionLevel : ModeBasedResolver.SUPPORTED_COMPRESSION_LEVELS) {
            String indexName = INDEX_NAME + compressionLevel;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            validateSearch(indexName, METHOD_PARAMETER_EF_SEARCH, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH);
        }

        for (CompressionLevel compressionLevel : ModeBasedResolver.SUPPORTED_COMPRESSION_LEVELS) {
            for (String mode : Mode.NAMES_ARRAY) {
                String indexName = INDEX_NAME + compressionLevel + "_" + mode;
                builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(FIELD_NAME)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .field(MODE_PARAMETER, mode)
                    .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
                    .endObject()
                    .endObject()
                    .endObject();
                String mapping = builder.toString();
                validateIndex(indexName, mapping);
                validateSearch(indexName, METHOD_PARAMETER_EF_SEARCH, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH);
            }
        }

        for (String mode : Mode.NAMES_ARRAY) {
            String indexName = INDEX_NAME + mode;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .field(MODE_PARAMETER, mode)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            validateSearch(indexName, METHOD_PARAMETER_EF_SEARCH, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH);
        }
    }

    @SneakyThrows
    public void testDeletedDocsWithSegmentMerge_whenValid_ThenSucceed() {
        XContentBuilder builder;
        CompressionLevel compressionLevel = CompressionLevel.x32;
        String indexName = INDEX_NAME + compressionLevel;
        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        validateIndexWithDeletedDocs(indexName, mapping);
        validateSearch(indexName, METHOD_PARAMETER_EF_SEARCH, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH);
    }

    @SneakyThrows
    public void testTraining_whenInvalid_thenFail() {
        setupTrainingIndex();
        String modelId = "test";
        XContentBuilder builder1 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
            .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .endObject()
            .field(MODEL_DESCRIPTION, "")
            .field(MODE_PARAMETER, Mode.ON_DISK)
            .endObject();
        expectThrows(ResponseException.class, () -> trainModel(modelId, builder1));

        XContentBuilder builder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
            .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .field(VECTOR_DATA_TYPE_FIELD, "binary")
            .field(MODEL_DESCRIPTION, "")
            .field(MODE_PARAMETER, Mode.ON_DISK)
            .endObject();
        expectThrows(ResponseException.class, () -> trainModel(modelId, builder2));
    }

    // Training isnt currently supported for mode and compression because quantization framework does not quantize
    // the training vectors. So, commenting out for now.
    @Ignore
    @SneakyThrows
    public void testTraining_whenValid_thenSucceed() {
        setupTrainingIndex();
        XContentBuilder builder;
        for (CompressionLevel compressionLevel : ModeBasedResolver.SUPPORTED_COMPRESSION_LEVELS) {
            String indexName = INDEX_NAME + compressionLevel;
            String modelId = indexName;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
                .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
                .field(KNNConstants.DIMENSION, DIMENSION)
                .field(MODEL_DESCRIPTION, "")
                .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
                .endObject();
            validateTraining(modelId, builder);
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("model_id", modelId)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            validateSearch(indexName, METHOD_PARAMETER_NPROBES, METHOD_PARAMETER_NLIST_DEFAULT);
        }

        for (CompressionLevel compressionLevel : ModeBasedResolver.SUPPORTED_COMPRESSION_LEVELS) {
            for (String mode : Mode.NAMES_ARRAY) {
                String indexName = INDEX_NAME + compressionLevel + "_" + mode;
                String modelId = indexName;
                builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
                    .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
                    .field(KNNConstants.DIMENSION, DIMENSION)
                    .field(MODEL_DESCRIPTION, "")
                    .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
                    .field(MODE_PARAMETER, mode)
                    .endObject();
                validateTraining(modelId, builder);
                builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(FIELD_NAME)
                    .field("type", "knn_vector")
                    .field("model_id", modelId)
                    .endObject()
                    .endObject()
                    .endObject();
                String mapping = builder.toString();
                validateIndex(indexName, mapping);
                validateSearch(indexName, METHOD_PARAMETER_NPROBES, METHOD_PARAMETER_NLIST_DEFAULT);
            }
        }

        for (String mode : Mode.NAMES_ARRAY) {
            String indexName = INDEX_NAME + mode;
            String modelId = indexName;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
                .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
                .field(KNNConstants.DIMENSION, DIMENSION)
                .field(MODEL_DESCRIPTION, "")
                .field(MODE_PARAMETER, mode)
                .endObject();
            validateTraining(modelId, builder);
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("model_id", modelId)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            validateSearch(indexName, METHOD_PARAMETER_NPROBES, METHOD_PARAMETER_NLIST_DEFAULT);
        }

    }

    @SneakyThrows
    private void validateIndex(String indexName, String mapping) {
        createKnnIndex(indexName, mapping);
        addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS);
        forceMergeKnnIndex(indexName, 1);
    }

    @SneakyThrows
    private void validateIndexWithDeletedDocs(String indexName, String mapping) {
        createKnnIndex(indexName, mapping);
        addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS);
        refreshIndex(indexName);
        // this will simulate the deletion of the docs
        addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS);
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName, 1);
        refreshIndex(indexName);
    }

    @SneakyThrows
    private void setupTrainingIndex() {
        createBasicKnnIndex(TRAINING_INDEX_NAME, TRAINING_FIELD_NAME, DIMENSION);
        bulkIngestRandomVectors(TRAINING_INDEX_NAME, TRAINING_FIELD_NAME, TRAINING_VECS, DIMENSION);
    }

    @SneakyThrows
    private void validateTraining(String modelId, XContentBuilder builder) {
        Response trainResponse = trainModel(modelId, builder);
        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
        assertTrainingSucceeds(modelId, 360, 1000);
    }

    @SneakyThrows
    private void validateSearch(String indexName, String methodParameterName, int methodParameterValue) {
        // Basic search
        Response response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", TEST_VECTOR)
                .field("k", K)
                .startObject(METHOD_PARAMETER)
                .field(methodParameterName, methodParameterValue)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        String responseBody = EntityUtils.toString(response.getEntity());
        List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());

        // Search with rescore
        response = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("knn")
                .startObject(FIELD_NAME)
                .field("vector", TEST_VECTOR)
                .field("k", K)
                .startObject(RescoreParser.RESCORE_PARAMETER)
                .field(RescoreParser.RESCORE_OVERSAMPLE_PARAMETER, 2.0f)
                .endObject()
                .startObject(METHOD_PARAMETER)
                .field(methodParameterName, methodParameterValue)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(response);
        responseBody = EntityUtils.toString(response.getEntity());
        knnResults = parseSearchResponse(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());
    }
}
