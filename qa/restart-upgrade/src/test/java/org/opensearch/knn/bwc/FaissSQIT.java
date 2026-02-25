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

package org.opensearch.knn.bwc;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.NAME;

public class FaissSQIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final String TRAIN_TEST_FIELD = "train-test-field";
    private static final String TRAIN_INDEX = "train-index";
    private static final String TEST_MODEL = "test-model";
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 100;

    public void testHNSWSQFP16_onUpgradeWhenIndexedAndQueried_thenSucceed() throws Exception {
        if (!isRunningAgainstOldCluster()) {
            SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.INNER_PRODUCT };
            Random random = new Random();
            SpaceType spaceType = spaceTypes[random.nextInt(spaceTypes.length)];

            List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
            List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
            List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

            // Create an index
            /**
             * "properties": {
             *     "test-field": {
             *         "type": "knn_vector",
             *         "dimension": 128,
             *         "method": {
             *             "name": "hnsw",
             *             "space_type": "l2",
             *             "engine": "faiss",
             *             "parameters": {
             *                 "m": 16,
             *                 "ef_construction": 128,
             *                 "ef_search": 128,
             *                 "encoder": {
             *                     "name": "sq",
             *                     "parameters": {
             *                        "type": "fp16"
             *                     }
             *                 }
             *             }
             *         }
             *     }
             * }
             */
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
                .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
                .startObject(KNNConstants.PARAMETERS)
                .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
                .field(
                    KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION,
                    efConstructionValues.get(random().nextInt(efConstructionValues.size()))
                )
                .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, ENCODER_SQ)
                .startObject(PARAMETERS)
                .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject();

            Map<String, Object> mappingMap = xContentBuilderToMap(builder);
            String mapping = builder.toString();

            createKnnIndex(testIndex, mapping);
            assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(testIndex)));
            indexTestData(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS);
            queryTestData(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS);
            deleteKNNIndex(testIndex);
            validateGraphEviction();
        }
    }

    public void testHNSWSQFP16_onUpgradeWhenClipToFp16isTrueAndIndexedWithOutOfFP16Range_thenSucceed() throws Exception {
        if (!isRunningAgainstOldCluster()) {

            List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
            List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);
            List<Integer> efSearchValues = ImmutableList.of(16, 32, 64, 128);

            int dimension = 128;

            // Create an index
            /**
             * "properties": {
             *     "test-field": {
             *         "type": "knn_vector",
             *         "dimension": 128,
             *         "method": {
             *             "name": "hnsw",
             *             "space_type": "l2",
             *             "engine": "faiss",
             *             "parameters": {
             *                 "m": 16,
             *                 "ef_construction": 128,
             *                 "ef_search": 128,
             *                 "encoder": {
             *                     "name": "sq",
             *                     "parameters": {
             *                        "type": "fp16",
             *                        "clip": true
             *                     }
             *                 }
             *             }
             *         }
             *     }
             * }
             */
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field("type", "knn_vector")
                .field("dimension", dimension)
                .startObject(KNNConstants.KNN_METHOD)
                .field(KNNConstants.NAME, KNNConstants.METHOD_HNSW)
                .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                .field(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName())
                .startObject(PARAMETERS)
                .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
                .field(
                    KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION,
                    efConstructionValues.get(random().nextInt(efConstructionValues.size()))
                )
                .field(KNNConstants.METHOD_PARAMETER_EF_SEARCH, efSearchValues.get(random().nextInt(efSearchValues.size())))
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, ENCODER_SQ)
                .startObject(PARAMETERS)
                .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
                .field(FAISS_SQ_CLIP, true)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject();

            Map<String, Object> mappingMap = xContentBuilderToMap(builder);
            String mapping = builder.toString();

            createKnnIndex(testIndex, mapping);
            assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(testIndex)));

            Float[] vector1 = new Float[dimension];
            Float[] vector2 = new Float[dimension];
            Float[] vector3 = new Float[dimension];
            Float[] vector4 = new Float[dimension];
            float[] queryVector = new float[dimension];
            int halfDimension = dimension / 2;

            for (int i = 0; i < dimension; i++) {
                if (i < halfDimension) {
                    vector1[i] = -65523.76f;
                    vector2[i] = -270.85f;
                    vector3[i] = -150.9f;
                    vector4[i] = -20.89f;
                    queryVector[i] = -10.5f;
                } else {
                    vector1[i] = 65504.2f;
                    vector2[i] = 65514.2f;
                    vector3[i] = 65504.0f;
                    vector4[i] = 100000000.0f;
                    queryVector[i] = 25.48f;
                }
            }

            addKnnDoc(testIndex, "1", TEST_FIELD, vector1);
            addKnnDoc(testIndex, "2", TEST_FIELD, vector2);
            addKnnDoc(testIndex, "3", TEST_FIELD, vector3);
            addKnnDoc(testIndex, "4", TEST_FIELD, vector4);

            int k = 4;
            Response searchResponse = searchKNNIndex(testIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, k), k);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), TEST_FIELD);
            assertEquals(k, results.size());
            for (int i = 0; i < k; i++) {
                assertEquals(k - i, Integer.parseInt(results.get(i).getDocId()));
            }
            deleteKNNIndex(testIndex);
            validateGraphEviction();
        }
    }

    public void testIVFSQFP16_onUpgradeWhenIndexedAndQueried_thenSucceed() throws Exception {
        if (!isRunningAgainstOldCluster()) {

            // Add training data
            createBasicKnnIndex(TRAIN_INDEX, TRAIN_TEST_FIELD, DIMENSION);
            int trainingDataCount = 1100;
            bulkIngestRandomVectors(TRAIN_INDEX, TRAIN_TEST_FIELD, trainingDataCount, DIMENSION);

            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .field(NAME, METHOD_IVF)
                .field(KNN_ENGINE, FAISS_NAME)
                .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
                .startObject(PARAMETERS)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, ENCODER_SQ)
                .startObject(PARAMETERS)
                .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            Map<String, Object> method = xContentBuilderToMap(builder);

            trainModel(TEST_MODEL, TRAIN_INDEX, TRAIN_TEST_FIELD, DIMENSION, method, "faiss ivf sqfp16 test description");

            // Make sure training succeeds after 30 seconds
            assertTrainingSucceeds(TEST_MODEL, 30, 1000);

            // Create knn index from model
            String indexMapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field("type", "knn_vector")
                .field(MODEL_ID, TEST_MODEL)
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), indexMapping);

            indexTestData(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS);
            queryTestData(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS);
            deleteKNNIndex(TRAIN_INDEX);
            deleteKNNIndex(testIndex);
            deleteModel(TEST_MODEL);
            validateGraphEviction();
        }
    }

    public void testIVFSQFP16_onUpgradeWhenClipToFp16isTrueAndIndexedWithOutOfFP16Range_thenSucceed() throws Exception {
        if (!isRunningAgainstOldCluster()) {
            int dimension = 2;

            // Add training data
            createBasicKnnIndex(TRAIN_INDEX, TRAIN_TEST_FIELD, dimension);
            int trainingDataCount = 1100;
            bulkIngestRandomVectors(TRAIN_INDEX, TRAIN_TEST_FIELD, trainingDataCount, dimension);

            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .field(NAME, METHOD_IVF)
                .field(KNN_ENGINE, FAISS_NAME)
                .field(METHOD_PARAMETER_SPACE_TYPE, "l2")
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_NLIST, 1)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, ENCODER_SQ)
                .startObject(PARAMETERS)
                .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
                .field(FAISS_SQ_CLIP, true)
                .endObject()
                .endObject()
                .endObject()
                .endObject();
            Map<String, Object> method = xContentBuilderToMap(builder);

            trainModel(TEST_MODEL, TRAIN_INDEX, TRAIN_TEST_FIELD, dimension, method, "faiss ivf sqfp16 test description");

            // Make sure training succeeds after 30 seconds
            assertTrainingSucceeds(TEST_MODEL, 30, 1000);

            // Create knn index from model
            String indexMapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field("type", "knn_vector")
                .field(MODEL_ID, TEST_MODEL)
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), indexMapping);
            Float[] vector1 = { -65523.76f, 65504.2f };
            Float[] vector2 = { -270.85f, 65514.2f };
            Float[] vector3 = { -150.9f, 65504.0f };
            Float[] vector4 = { -20.89f, 100000000.0f };
            addKnnDoc(testIndex, "1", TEST_FIELD, vector1);
            addKnnDoc(testIndex, "2", TEST_FIELD, vector2);
            addKnnDoc(testIndex, "3", TEST_FIELD, vector3);
            addKnnDoc(testIndex, "4", TEST_FIELD, vector4);

            float[] queryVector = { -10.5f, 25.48f };
            int k = 4;
            Response searchResponse = searchKNNIndex(testIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, k), k);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), TEST_FIELD);
            assertEquals(k, results.size());
            for (int i = 0; i < k; i++) {
                assertEquals(k - i, Integer.parseInt(results.get(i).getDocId()));
            }

            deleteKNNIndex(testIndex);
            deleteKNNIndex(TRAIN_INDEX);
            deleteModel(TEST_MODEL);
            validateGraphEviction();
        }
    }

    private void validateGraphEviction() throws Exception {
        // Search every 5 seconds 14 times to confirm graph gets evicted
        int intervals = 14;
        for (int i = 0; i < intervals; i++) {
            if (getTotalGraphsInCache() == 0) {
                return;
            }

            Thread.sleep(5 * 1000);
        }

        fail("Graphs are not getting evicted");
    }

    private void queryTestData(final String indexName, final String fieldName, final int dimension, final int numDocs) throws IOException,
        ParseException {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);
        int k = 10;

        Response searchResponse = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, queryVector, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
        }
    }

    private void indexTestData(final String indexName, final String fieldName, final int dimension, final int numDocs) throws Exception {
        for (int i = 0; i < numDocs; i++) {
            float[] indexVector = new float[dimension];
            Arrays.fill(indexVector, (float) i);
            addKnnDocWithAttributes(indexName, Integer.toString(i), fieldName, indexVector, ImmutableMap.of("rating", String.valueOf(i)));
        }

        // Assert that all docs are ingested
        refreshAllNonSystemIndices();
        assertEquals(numDocs, getDocCount(indexName));
    }

}
