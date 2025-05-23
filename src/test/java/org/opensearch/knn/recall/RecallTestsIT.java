/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.recall;

import lombok.SneakyThrows;
import org.junit.Before;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.annotation.ExpectRemoteBuildValidation;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_CODE_SIZE;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PARAMETER_PQ_M;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
// import static org.opensearch.knn.common.KNNConstants.ENCODER_BINARY;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY;
import static org.opensearch.knn.index.KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;

/**
 * Tests confirm that for the different supported configurations, recall is sound. The recall thresholds are
 * conservatively and empirically determined to prevent flakiness.
 *
 * This test suite can take a long time to run. The primary reason is that training can take a long time for PQ.
 * The parameters for PQ have been reduced significantly, but it still takes time.
 */
public class RecallTestsIT extends KNNRestTestCase {

    private static final String PROPERTIES_FIELD = "properties";
    private final static String TEST_INDEX_PREFIX_NAME = "test_index";
    private final static String TEST_FIELD_NAME = "test_field";
    private final static String TRAIN_INDEX_NAME = "train_index";
    private final static String TRAIN_FIELD_NAME = "train_field";
    private final static String TEST_MODEL_ID = "test_model_id";
    private final static int TEST_DIMENSION = 32;
    private final static int DOC_COUNT = 1100;
    private final static int QUERY_COUNT = 100;
    private final static int TEST_K = 100;
    private final static double PERFECT_RECALL = 1.0;
    private final static int SHARD_COUNT = 1;
    private final static int REPLICA_COUNT = 0;
    private final static int MAX_SEGMENT_COUNT = 10;

    // Standard algorithm parameters
    private final static int HNSW_M = 16;
    private final static int HNSW_EF_CONSTRUCTION = 100;
    private final static int HNSW_EF_SEARCH = TEST_K; // For consistency with lucene
    private final static int IVF_NLIST = 4;
    private final static int IVF_NPROBES = IVF_NLIST; // This equates to essentially a brute force search
    private final static int PQ_CODE_SIZE = 8; // This is low and going to produce bad recall, but reduces build time
    private final static int PQ_M = TEST_DIMENSION / 8; // Will give low recall, but required for test time

    // on disk settings
    private final static String ON_DISK = "on_disk";

    // Setup ground truth for all tests once
    private final static float[][] INDEX_VECTORS = TestUtils.getIndexVectors(DOC_COUNT, TEST_DIMENSION, true);
    private final static float[][] QUERY_VECTORS = TestUtils.getQueryVectors(QUERY_COUNT, TEST_DIMENSION, DOC_COUNT, true);
    private final static Map<SpaceType, List<Set<String>>> GROUND_TRUTH = Map.of(
        SpaceType.L2,
        TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.L2, TEST_K),
        SpaceType.COSINESIMIL,
        TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.COSINESIMIL, TEST_K),
        SpaceType.INNER_PRODUCT,
        TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.INNER_PRODUCT, TEST_K)
    );

    @SneakyThrows
    @Before
    public void setupClusterSettings() {
        updateClusterSettings(KNN_ALGO_PARAM_INDEX_THREAD_QTY, 2);
        updateClusterSettings(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, true);
    }

    /**
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {TEST_DIMENSION},
     *      "mode": "on_disk"
     *      "method": {
     *          "name":"hnsw",
     *          "engine":"faiss",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION},
     *              "ef_search": {HNSW_EF_SEARCH},
     *              "encoder": {
     *                  "name": "binary",
     *                  "parameters": {
     *                      "bits": {num_bits},
     *                      "random_rotation": true
     *                  }
     *              }
     *          }
     *       }
     *     }
     *   }
     *
     * }
     * Recall values for seed F71F949325FE8B42:
     * (note that the vectors in these tests are randomly sampled from a gaussian distribution.
     * It's expected that random rotation does not outperform normal search in this case, and we only see benefits for
     * non-random data with uneven variance across dimensions.)
     * With Random Rotation ON:
     * L2:
     *     1 bit:  0.8773 recall
     *     2 bits: 0.9451 recall
     *     4 bits: 0.9678 recall
     *
     * INNER_PRODUCT:
     *     1 bit:  0.6817 recall
     *     2 bits: 0.7220 recall
     *     4 bits: 0.7341 recall
     *
     * COSINE:
     *     1 bit:  0.8710 recall
     *     2 bits: 0.9416 recall
     *     4 bits: 0.9700 recall
     *
     * With Random Rotation OFF:
     * L2:
     *     1 bit:  0.9208 recall
     *     2 bits: 0.9746 recall
     *     4 bits: 0.9924 recall
     *
     * INNER_PRODUCT:
     *     1 bit:  0.7106 recall
     *     2 bits: 0.7356 recall
     *     4 bits: 0.7457 recall
     *
     * COSINE:
     *     1 bit:  0.9107 recall
     *     2 bits: 0.9677 recall
     *     4 bits: 0.9860 recall
     *
     * Test duration: ~1 minute locally (m3 pro)
     */
    @SneakyThrows
    public void testRecall_whenRandomRotationEnabled_thenRecallAbove60Percent() {
        List<SpaceType> spaceTypes = List.of(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);
        List<Integer> numBits = List.of(1, 2, 4);
        for (SpaceType spaceType : spaceTypes) {
            for (Integer bits : numBits) {
                String indexName = createIndexName(KNNEngine.FAISS, spaceType) + "(" + bits + "bits)";
                XContentBuilder builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES_FIELD)
                    .startObject(TEST_FIELD_NAME)
                    .field(TYPE, TYPE_KNN_VECTOR)
                    .field(DIMENSION, TEST_DIMENSION)
                    .field(MODE_PARAMETER, ON_DISK)
                    .startObject(KNN_METHOD)
                    .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                    .field(KNN_ENGINE, KNNEngine.FAISS.getName())
                    .field(NAME, METHOD_HNSW)
                    .startObject(PARAMETERS)
                    .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
                    .field(METHOD_PARAMETER_M, HNSW_M)
                    .field(METHOD_PARAMETER_EF_SEARCH, 100)
                    .startObject("encoder")
                    .field(NAME, "binary")
                    .startObject("parameters")
                    .field("bits", (int) bits)
                    .field(QFrameBitEncoder.ENABLE_RANDOM_ROTATION_PARAM, true)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject();
                createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), builder.toString());
                assertRecall(indexName, spaceType, 0.4f);
            }
        }
    }

    /**
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {DIMENSION},
     *      "method": {
     *          "name":"hnsw",
     *          "engine":"lucene",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION}
     *          }
     *       }
     *     }
     *   }
     * }
     */
    @SneakyThrows
    public void testRecall_whenLuceneHnswFP32_thenRecallAbove75percent() {
        List<SpaceType> spaceTypes = List.of(SpaceType.L2, SpaceType.COSINESIMIL);
        for (SpaceType spaceType : spaceTypes) {
            String indexName = createIndexName(KNNEngine.LUCENE, spaceType);
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES_FIELD)
                .startObject(TEST_FIELD_NAME)
                .field(TYPE, TYPE_KNN_VECTOR)
                .field(DIMENSION, TEST_DIMENSION)
                .startObject(KNN_METHOD)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
                .field(METHOD_PARAMETER_M, HNSW_M)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), builder.toString());
            assertRecall(indexName, spaceType, 0.25f);
        }
    }

    /**
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {TEST_DIMENSION},
     *      "method": {
     *          "name":"hnsw",
     *          "engine":"faiss",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION},
     *              "ef_search": {HNSW_EF_SEARCH},
     *          }
     *       }
     *     }
     *   }
     * }
     */
    @SneakyThrows
    @ExpectRemoteBuildValidation
    public void testRecall_whenFaissHnswFP32_thenRecallAbove75percent() {
        List<SpaceType> spaceTypes = List.of(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);
        for (SpaceType spaceType : spaceTypes) {
            String indexName = createIndexName(KNNEngine.FAISS, spaceType);
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES_FIELD)
                .startObject(TEST_FIELD_NAME)
                .field(TYPE, TYPE_KNN_VECTOR)
                .field(DIMENSION, TEST_DIMENSION)
                .startObject(KNN_METHOD)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .field(KNN_ENGINE, KNNEngine.FAISS.getName())
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
                .field(METHOD_PARAMETER_M, HNSW_M)
                .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), builder.toString());
            assertRecall(indexName, spaceType, 0.25f);
        }
    }

    /**
     * Train context:
     * {
     *  "method": {
     *      "name":"ivf",
     *      "engine":"faiss",
     *      "space_type": "{SPACE_TYPE}",
     *      "parameters":{
     *          "nlist":{IVF_NLIST},
     *          "nprobes": {IVF_NPROBES}
     *      }
     *   }
     * }
     *
     * Index Mapping:
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "model_id": {MODEL_ID}
     *     }
     *   }
     * }
     */
    @SneakyThrows
    public void testRecall_whenFaissIVFFP32_thenRecallAbove75percent() {
        List<SpaceType> spaceTypes = List.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        setupTrainingIndex();
        for (SpaceType spaceType : spaceTypes) {
            String indexName = createIndexName(KNNEngine.FAISS, spaceType);

            // Train the model
            XContentBuilder trainingBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .field(NAME, METHOD_IVF)
                .field(KNN_ENGINE, FAISS_NAME)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_NLIST, IVF_NLIST)
                .field(METHOD_PARAMETER_NPROBES, IVF_NPROBES)
                .endObject()
                .endObject();
            trainModel(
                TEST_MODEL_ID,
                TRAIN_INDEX_NAME,
                TRAIN_FIELD_NAME,
                TEST_DIMENSION,
                xContentBuilderToMap(trainingBuilder),
                String.format("%s-%s", KNNEngine.FAISS.getName(), spaceType.getValue())
            );
            assertTrainingSucceeds(TEST_MODEL_ID, 100, 1000 * 5);

            // Build the index
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), getModelMapping());
            assertRecall(indexName, spaceType, 0.25f);

            deleteIndex(indexName);

            // Delete the model
            deleteModel(TEST_MODEL_ID);
        }
    }

    /**
     * Train context:
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {TEST_DIMENSION},
     *      "method": {
     *          "name":"hnsw",
     *          "engine":"faiss",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION},
     *              "ef_search": {HNSW_EF_SEARCH},
     *          }
     *       }
     *     }
     *   }
     * }
     *
     * Index Mapping:
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "model_id": {MODEL_ID}
     *     }
     *   }
     * }
     */
    @SneakyThrows
    public void testRecall_whenFaissIVFPQFP32_thenRecallAbove50percent() {
        List<SpaceType> spaceTypes = List.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        setupTrainingIndex();
        for (SpaceType spaceType : spaceTypes) {
            String indexName = createIndexName(KNNEngine.FAISS, spaceType);

            // Train the model
            XContentBuilder trainingBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .field(NAME, METHOD_IVF)
                .field(KNN_ENGINE, FAISS_NAME)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_NLIST, IVF_NLIST)
                .field(METHOD_PARAMETER_NPROBES, IVF_NPROBES)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, ENCODER_PQ)
                .startObject(PARAMETERS)
                .field(ENCODER_PARAMETER_PQ_CODE_SIZE, PQ_CODE_SIZE)
                .field(ENCODER_PARAMETER_PQ_M, PQ_M)
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            trainModel(
                TEST_MODEL_ID,
                TRAIN_INDEX_NAME,
                TRAIN_FIELD_NAME,
                TEST_DIMENSION,
                xContentBuilderToMap(trainingBuilder),
                String.format("%s-%s", KNNEngine.FAISS.getName(), spaceType.getValue())
            );
            assertTrainingSucceeds(TEST_MODEL_ID, 100, 1000 * 5);

            // Build the index
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), getModelMapping());
            assertRecall(indexName, spaceType, 0.5f);

            deleteIndex(indexName);

            // Delete the model
            deleteModel(TEST_MODEL_ID);
        }
    }

    /**
     * Train context:
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {TEST_DIMENSION},
     *      "method": {
     *          "name":"hnsw",
     *          "engine":"faiss",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION},
     *              "ef_search": {HNSW_EF_SEARCH},
     *          }
     *       }
     *     }
     *   }
     * }
     *
     * Index Mapping:
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "model_id": {MODEL_ID}
     *     }
     *   }
     * }
     */
    @SneakyThrows
    public void testRecall_whenFaissHNSWPQFP32_thenRecallAbove50percent() {
        List<SpaceType> spaceTypes = List.of(SpaceType.L2, SpaceType.INNER_PRODUCT);
        setupTrainingIndex();
        for (SpaceType spaceType : spaceTypes) {
            String indexName = createIndexName(KNNEngine.FAISS, spaceType);

            // Train the model
            XContentBuilder trainingBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .field(NAME, METHOD_HNSW)
                .field(KNN_ENGINE, FAISS_NAME)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_M, HNSW_M)
                .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, ENCODER_PQ)
                .startObject(PARAMETERS)
                .field(ENCODER_PARAMETER_PQ_CODE_SIZE, PQ_CODE_SIZE)
                .field(ENCODER_PARAMETER_PQ_M, PQ_M)
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            trainModel(
                TEST_MODEL_ID,
                TRAIN_INDEX_NAME,
                TRAIN_FIELD_NAME,
                TEST_DIMENSION,
                xContentBuilderToMap(trainingBuilder),
                String.format("%s-%s", KNNEngine.FAISS.getName(), spaceType.getValue())
            );
            assertTrainingSucceeds(TEST_MODEL_ID, 100, 1000 * 5);

            // Build the index
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), getModelMapping());
            assertRecall(indexName, spaceType, 0.5f);

            deleteIndex(indexName);

            // Delete the model
            deleteModel(TEST_MODEL_ID);
        }
    }

    @SneakyThrows
    private void assertRecall(String testIndexName, SpaceType spaceType, float acceptableRecallFromPerfect) {
        List<List<String>> searchResults = bulkSearch(testIndexName, TEST_FIELD_NAME, QUERY_VECTORS, TEST_K);
        logger.info("bulk search complete");
        double recallValue = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH.get(spaceType), TEST_K);
        logger.info("Recall value for SpaceType {} = {}", spaceType, recallValue);
        assertEquals(PERFECT_RECALL, recallValue, acceptableRecallFromPerfect);
    }

    private String createIndexName(KNNEngine knnEngine, SpaceType spaceType) {
        return String.format("%s_%s_%s", TEST_INDEX_PREFIX_NAME, knnEngine.getName(), spaceType.getValue());
    }

    @SneakyThrows
    private void createIndexAndIngestDocs(String indexName, String fieldName, Settings settings, String mapping) {
        createKnnIndex(indexName, settings, mapping);
        bulkAddKnnDocs(indexName, fieldName, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, MAX_SEGMENT_COUNT);
    }

    @SneakyThrows
    private void setupTrainingIndex() {
        XContentBuilder trainingIndexBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(TRAIN_FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, TEST_DIMENSION)
            .endObject()
            .endObject()
            .endObject();
        createIndexAndIngestDocs(
            TRAIN_INDEX_NAME,
            TRAIN_FIELD_NAME,
            Settings.builder().put("number_of_shards", SHARD_COUNT).put("number_of_replicas", REPLICA_COUNT).build(),
            trainingIndexBuilder.toString()
        );
    }

    @SneakyThrows
    private String getModelMapping() {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(TEST_FIELD_NAME)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(MODEL_ID, TEST_MODEL_ID)
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    private Settings getSettings() {
        return Settings.builder()
            .put("number_of_shards", SHARD_COUNT)
            .put("number_of_replicas", REPLICA_COUNT)
            .put("index.knn", true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();
    }
}
