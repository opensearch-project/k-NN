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
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.rest.RestStatus;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ScriptScoringIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 3;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    // KNN script scoring for space_type "l2"
    public void testKNNL2ScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K, SpaceType.L2);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K, SpaceType.L2);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "l1"
    public void testKNNL1ScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K, SpaceType.L1);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K, SpaceType.L1);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "linf"
    public void testKNNLinfScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K, SpaceType.LINF);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K, SpaceType.LINF);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "innerproduct"
    public void testKNNInnerProductScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            validateKNNInnerProductScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
            validateKNNInnerProductScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K);
            deleteKNNIndex(testIndex);
        }
    }

    // Validate Script score search for space_type : "inner_product"
    private void validateKNNInnerProductScriptScoreSearch(String testIndex, String testField, int dimension, int numDocs, int k)
        throws Exception {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);

        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        params.put("field", testField);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.INNER_PRODUCT.getValue());

        Request request = constructKNNScriptQueryRequest(testIndex, qb, params, k);
        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), testField);
        assertEquals(k, results.size());

        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
        }
    }

}
