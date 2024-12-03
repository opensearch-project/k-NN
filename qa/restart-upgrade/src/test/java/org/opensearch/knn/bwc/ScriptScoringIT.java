/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.http.util.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.IDVectorProducer;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.TestUtils.FIELD;
import static org.opensearch.knn.TestUtils.KNN_BWC_PREFIX;
import static org.opensearch.knn.TestUtils.QUERY_VALUE;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class ScriptScoringIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 3;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static int QUERY_COUNT = 0;
    private static final String TRAINING_INDEX_DEFAULT = KNN_BWC_PREFIX + "train-index-default-1";
    private static final String TRAINING_FIELD = "train-field";
    private static final String TEST_MODEL_ID_DEFAULT = "test-model-id-default-1";
    private static final String MODEL_DESCRIPTION = "Description for train model test";

    // KNN script scoring for space_type "l2"
    public void testKNNL2ScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            QUERY_COUNT = NUM_DOCS;
            DOC_ID = NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "l1"
    public void testKNNL1ScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            QUERY_COUNT = NUM_DOCS;
            DOC_ID = NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L1);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L1);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "linf"
    public void testKNNLinfScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            QUERY_COUNT = NUM_DOCS;
            DOC_ID = NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.LINF);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.LINF);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "innerproduct"
    public void testKNNInnerProductScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            QUERY_COUNT = NUM_DOCS;
            DOC_ID = NUM_DOCS;
            validateKNNInnerProductScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            validateKNNInnerProductScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
            deleteKNNIndex(testIndex);
        }
    }

    // Validate Script score search for space_type : "inner_product"
    private void validateKNNInnerProductScriptScoreSearch(String testIndex, String testField, int dimension, int numDocs, int k)
        throws Exception {
        IDVectorProducer idVectorProducer = new IDVectorProducer(dimension, numDocs);
        float[] queryVector = idVectorProducer.getVector(numDocs);

        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        params.put(FIELD, testField);
        params.put(QUERY_VALUE, queryVector);
        params.put(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue());

        Request request = constructKNNScriptQueryRequest(testIndex, qb, params, k, Collections.emptyMap());
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), testField);
        assertEquals(k, results.size());

        for (int i = 0; i < k; i++) {
            int expDocID = numDocs - i - 1;
            int actualDocID = Integer.parseInt(results.get(i).getDocId());
            assertEquals(expDocID, actualDocID);
        }
    }

    public void testNonKNNIndex_withMethodParams() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                createKNNDefaultScriptScoreSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, "hnsw", KNNEngine.FAISS.getName(), SpaceType.DEFAULT.getValue(), false)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            QUERY_COUNT = NUM_DOCS;
            DOC_ID = NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
            deleteKNNIndex(testIndex);
        }
    }

    public void testNonKNNIndex_withModelId() throws Exception {
        if (isRunningAgainstOldCluster()) {
            // Create a training index and randomly ingest data into it
            createBasicKnnIndex(TRAINING_INDEX_DEFAULT, TRAINING_FIELD, DIMENSIONS);
            bulkIngestRandomVectors(TRAINING_INDEX_DEFAULT, TRAINING_FIELD, NUM_DOCS, DIMENSIONS);

            trainKNNModel(TEST_MODEL_ID_DEFAULT, TRAINING_INDEX_DEFAULT, TRAINING_FIELD, DIMENSIONS, MODEL_DESCRIPTION);
            validateModelCreated(TEST_MODEL_ID_DEFAULT);

            createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, TEST_MODEL_ID_DEFAULT));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            QUERY_COUNT = NUM_DOCS;
            DOC_ID = NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
            deleteKNNIndex(testIndex);
        }
    }

    // train KNN model
    // method : "ivf", engine : "faiss", space_type : "l2", nlists : 1
    public void trainKNNModel(String modelId, String trainingIndexName, String trainingFieldName, int dimension, String description)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        Response trainResponse = trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, description);
        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
    }

    // Confirm that the model gets created using Get Model API
    public void validateModelCreated(String modelId) throws Exception {
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();
        assertEquals(modelId, responseMap.get(MODEL_ID));
        assertTrainingSucceeds(modelId, NUM_OF_ATTEMPTS, DELAY_MILLI_SEC);
    }

}
