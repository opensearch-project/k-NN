/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Assert;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.parser.RescoreParser;

import java.util.List;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST_DEFAULT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TRAIN_FIELD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.TRAIN_INDEX_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.MAPPING_COMPRESSION_NAMES_ARRAY;

public class ModeAndCompressionIT extends KNNRestTestCase {

    private static final String TRAINING_INDEX_NAME = "training_index";
    private static final String TRAINING_FIELD_NAME = "training_field";
    private static final int TRAINING_VECS = 1100;

    private static final int DIMENSION = 16;
    private static final int NUM_DOCS = 20;
    private static final int K = NUM_DOCS;
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

    private static final String[] COMPRESSION_LEVELS = new String[] {
        CompressionLevel.x2.getName(),
        CompressionLevel.x4.getName(),
        CompressionLevel.x8.getName(),
        CompressionLevel.x16.getName(),
        CompressionLevel.x32.getName() };

    @SneakyThrows
    public void testIndexCreation_whenInvalid_thenFail() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
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
        for (String compressionLevel : COMPRESSION_LEVELS) {
            String indexName = INDEX_NAME + compressionLevel;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
                .endObject()
                .endObject()
                .endObject();
            String mapping = builder.toString();
            validateIndex(indexName, mapping);
            logger.info("Compression level {}", compressionLevel);
            validateSearch(
                indexName,
                METHOD_PARAMETER_EF_SEARCH,
                KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                compressionLevel,
                Mode.NOT_CONFIGURED.getName()
            );
        }

        for (String compressionLevel : COMPRESSION_LEVELS) {
            for (String mode : Mode.NAMES_ARRAY) {
                String indexName = INDEX_NAME + compressionLevel + "_" + mode;
                builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(FIELD_NAME)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .field(MODE_PARAMETER, mode)
                    .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
                    .endObject()
                    .endObject()
                    .endObject();
                String mapping = builder.toString();
                validateIndex(indexName, mapping);
                logger.info("Compression level {}", compressionLevel);
                validateSearch(
                    indexName,
                    METHOD_PARAMETER_EF_SEARCH,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                    compressionLevel,
                    mode
                );
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
            logger.info("Compression level {}", CompressionLevel.NOT_CONFIGURED.getName());
            validateSearch(
                indexName,
                METHOD_PARAMETER_EF_SEARCH,
                KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                CompressionLevel.NOT_CONFIGURED.getName(),
                mode
            );
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
        validateGreenIndex(indexName);
    }

    @SneakyThrows
    public void testCompressionIndexWithNonVectorFieldsSegment_whenValid_ThenSucceed() {
        CompressionLevel compressionLevel = CompressionLevel.x32;
        String indexName = INDEX_NAME + compressionLevel;
        try (
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
                .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                .endObject()
                .endObject()
                .endObject()
        ) {
            String mapping = builder.toString();
            Settings indexSettings = buildKNNIndexSettings(0);
            createKnnIndex(indexName, indexSettings, mapping);
            // since we are going to delete a document, so its better to have 1 more extra doc so that we can re-use some tests
            addKNNDocs(indexName, FIELD_NAME, DIMENSION, 0, NUM_DOCS + 1);
            addNonKNNDoc(indexName, String.valueOf(NUM_DOCS + 2), FIELD_NAME_NON_KNN, "Hello world");
            deleteKnnDoc(indexName, "0");
            validateGreenIndex(indexName);
            validateSearch(
                indexName,
                METHOD_PARAMETER_EF_SEARCH,
                KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
                compressionLevel.getName(),
                Mode.ON_DISK.getName()
            );
        }
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
            .field(VECTOR_DATA_TYPE_FIELD, "float")
            .field(MODEL_DESCRIPTION, "")
            .field(MODE_PARAMETER, Mode.ON_DISK)
            .field(COMPRESSION_LEVEL_PARAMETER, "16x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, "pq")
            .startObject(PARAMETERS)
            .field(ENCODER_PARAMETER_PQ_CODE_SIZE, 2)
            .field(ENCODER_PARAMETER_PQ_M, 8)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
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

    @SneakyThrows
    public void testTraining_whenValid_thenSucceed() {
        setupTrainingIndex();
        XContentBuilder builder;
        for (String compressionLevel : MAPPING_COMPRESSION_NAMES_ARRAY) {
            if (compressionLevel.equals("4x")) {
                continue;
            }
            String indexName = INDEX_NAME + compressionLevel;
            String modelId = indexName;
            builder = XContentFactory.jsonBuilder()
                .startObject()
                .field(TRAIN_INDEX_PARAMETER, TRAINING_INDEX_NAME)
                .field(TRAIN_FIELD_PARAMETER, TRAINING_FIELD_NAME)
                .field(KNNConstants.DIMENSION, DIMENSION)
                .field(MODEL_DESCRIPTION, "")
                .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel)
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
            validateSearch(
                indexName,
                METHOD_PARAMETER_NPROBES,
                METHOD_PARAMETER_NLIST_DEFAULT,
                compressionLevel,
                Mode.NOT_CONFIGURED.getName()
            );
            deleteKNNIndex(indexName);
        }
        for (String mode : Mode.NAMES_ARRAY) {
            if (mode == null) {
                continue;
            }
            mode = mode.toLowerCase();
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
            validateSearch(
                indexName,
                METHOD_PARAMETER_NPROBES,
                METHOD_PARAMETER_NLIST_DEFAULT,
                CompressionLevel.NOT_CONFIGURED.getName(),
                mode
            );
            deleteKNNIndex(indexName);
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
    private void validateGreenIndex(String indexName) {
        Request request = new Request("GET", "/_cat/indices/" + indexName + "?format=csv");
        Response response = client().performRequest(request);
        assertOK(response);
        assertEquals(
            "The status of index " + indexName + " is not green",
            "green",
            new String(response.getEntity().getContent().readAllBytes()).split("\n")[0].split(" ")[0]
        );

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
    private void validateSearch(
        String indexName,
        String methodParameterName,
        int methodParameterValue,
        String compressionLevelString,
        String mode
    ) {
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
        List<Float> knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());

        // Do exact search and gather right scores for the documents
        Response exactSearchResponse = searchKNNIndex(
            indexName,
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("script_score")
                .startObject("query")
                .field("match_all")
                .startObject()
                .endObject()
                .endObject()
                .startObject("script")
                .field("source", "knn_score")
                .field("lang", "knn")
                .startObject("params")
                .field("field", FIELD_NAME)
                .field("query_value", TEST_VECTOR)
                .field("space_type", SpaceType.L2.getValue())
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject(),
            K
        );
        assertOK(exactSearchResponse);
        String exactSearchResponseBody = EntityUtils.toString(exactSearchResponse.getEntity());
        List<Float> exactSearchKnnResults = parseSearchResponseScore(exactSearchResponseBody, FIELD_NAME);
        assertEquals(NUM_DOCS, exactSearchKnnResults.size());

        if (CompressionLevel.x4.getName().equals(compressionLevelString) == false && Mode.ON_DISK.getName().equals(mode)) {
            Assert.assertEquals(exactSearchKnnResults, knnResults);
        }

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
        knnResults = parseSearchResponseScore(responseBody, FIELD_NAME);
        assertEquals(K, knnResults.size());
        if (CompressionLevel.x4.getName().equals(compressionLevelString) == false && Mode.ON_DISK.getName().equals(mode)) {
            Assert.assertEquals(exactSearchKnnResults, knnResults);
        }
    }
}
