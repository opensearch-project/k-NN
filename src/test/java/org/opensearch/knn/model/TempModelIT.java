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

package org.opensearch.knn.model;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.AfterClass;
import org.junit.FixMethodOrder;
import org.junit.runners.MethodSorters;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.TestUtils.KNN_BWC_PREFIX;
import static org.opensearch.knn.common.KNNConstants.*;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class TempModelIT extends KNNRestTestCase {

    private static final String TEST_MODEL_INDEX = KNN_BWC_PREFIX + "test-model-index";
    private static final String TEST_MODEL_INDEX_DEFAULT = KNN_BWC_PREFIX + "test-model-index-default";
    private static final String TRAINING_INDEX = KNN_BWC_PREFIX + "train-index";
    private static final String TRAINING_INDEX_DEFAULT = KNN_BWC_PREFIX + "train-index-default";
    private static final String TRAINING_FIELD = "train-field";
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static int DOC_ID_TEST_MODEL_INDEX = 0;
    private static int DOC_ID_TEST_MODEL_INDEX_DEFAULT = 0;
    private static final int DELAY_MILLI_SEC = 1000;
    private static final int EXP_NUM_OF_MODELS = 3;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static final int NUM_DOCS_TEST_MODEL_INDEX = 100;
    private static final int NUM_DOCS_TEST_MODEL_INDEX_DEFAULT = 100;
    private static final int NUM_OF_ATTEMPTS = 1;
    private static int QUERY_COUNT = 0;
    private static int QUERY_COUNT_TEST_MODEL_INDEX = 0;
    private static int QUERY_COUNT_TEST_MODEL_INDEX_DEFAULT = 0;
    private static final String TEST_MODEL_ID = "test-model-id";
    private static final String TEST_MODEL_ID_DEFAULT = "test-model-id-default";
    private static final String TEST_MODEL_ID_TRAINING = "test-model-id-training";
    private static final String MODEL_DESCRIPTION = "Description for train model test";

    public void testStartTraining() throws Exception {
        System.setProperty("tests.skip_delete_model_index", "true");
        createBasicKnnIndex(TRAINING_INDEX_DEFAULT, TRAINING_FIELD, DIMENSIONS);
        bulkIngestRandomVectors(TRAINING_INDEX_DEFAULT, TRAINING_FIELD, NUM_DOCS, DIMENSIONS);

        trainKNNModel(TEST_MODEL_ID_DEFAULT, TRAINING_INDEX_DEFAULT, TRAINING_FIELD, DIMENSIONS, MODEL_DESCRIPTION);
    }

    public void testAfterTraining() throws Exception {
        validateModelCreated(TEST_MODEL_ID_DEFAULT);
    }

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

    public void validateModelCreated(String modelId) throws Exception {
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        assertEquals(modelId, responseMap.get(MODEL_ID));
    }

    @Override
    protected boolean preserveClusterUponCompletion() {
        return false;
    }
}
