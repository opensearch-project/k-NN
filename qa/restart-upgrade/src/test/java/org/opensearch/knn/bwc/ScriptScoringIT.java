/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.http.util.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.IDVectorProducer;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.core.rest.RestStatus;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.TestUtils.FIELD;
import static org.opensearch.knn.TestUtils.QUERY_VALUE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;

public class ScriptScoringIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 3;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static int QUERY_COUNT = 0;

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

}
