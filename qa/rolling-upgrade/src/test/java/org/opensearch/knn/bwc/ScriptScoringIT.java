/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.client.Response;
import org.opensearch.knn.index.SpaceType;

import java.util.List;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class ScriptScoringIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    // KNN script scoring for space_type "l2"
    public void testKNNL2ScriptScore() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                int docIdOld = 0;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                break;
            case MIXED:
                int totalDocsCountMixed;
                int docIdMixed;
                if (isFirstMixedRound()) {
                    totalDocsCountMixed = NUM_DOCS;
                    docIdMixed = NUM_DOCS;
                } else {
                    totalDocsCountMixed = 2 * NUM_DOCS;
                    docIdMixed = 2 * NUM_DOCS;
                }
                validateKNNL2ScriptScoreOnUpgrade(totalDocsCountMixed, docIdMixed);
                break;
            case UPGRADED:
                int totalDocsCountUpgraded = 3 * NUM_DOCS;
                int docIdUpgraded = 3 * NUM_DOCS;
                validateKNNL2ScriptScoreOnUpgrade(totalDocsCountUpgraded, docIdUpgraded);

                deleteKNNIndex(testIndex);
        }
    }

    // validation steps for L2 script scoring after upgrading each node from old version to new version
    public void validateKNNL2ScriptScoreOnUpgrade(int totalDocsCount, int docId) throws Exception {
        validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, totalDocsCount, K, SpaceType.L2);
        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);

        totalDocsCount = totalDocsCount + NUM_DOCS;
        validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, totalDocsCount, K, SpaceType.L2);
    }

    // KNN script scoring for space_type "cosine"
    public void testKNNCosineScriptScore() throws Exception {
        float[] indexVector1 = { 1.1f, 2.1f, 3.3f };
        float[] indexVector2 = { 8.1f, 9.1f, 10.3f };
        float[] indexVector3 = { 9.1f, 10.1f, 11.3f };
        float[] indexVector4 = { 10.1f, 11.1f, 12.3f };
        float[] queryVector = { 3.0f, 4.0f, 13.5f };
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        int k = 10;
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, 3));
                addKnnDoc(testIndex, "1", TEST_FIELD, indexVector1);
                validateScore(k, queryVector, new float[] { cosineSimilarity(queryVector, indexVector1) });
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    addKnnDoc(testIndex, "2", TEST_FIELD, indexVector2);
                    validateScore(
                        k,
                        queryVector,
                        new float[] { cosineSimilarity(queryVector, indexVector1), cosineSimilarity(queryVector, indexVector2) }
                    );
                } else {
                    addKnnDoc(testIndex, "3", TEST_FIELD, indexVector3);
                    validateScore(
                        k,
                        queryVector,
                        new float[] {
                            cosineSimilarity(queryVector, indexVector1),
                            cosineSimilarity(queryVector, indexVector2),
                            cosineSimilarity(queryVector, indexVector3) }
                    );
                }
                break;
            case UPGRADED:
                addKnnDoc(testIndex, "4", TEST_FIELD, indexVector3);
                validateScore(
                    k,
                    queryVector,
                    new float[] {
                        cosineSimilarity(queryVector, indexVector1),
                        cosineSimilarity(queryVector, indexVector2),
                        cosineSimilarity(queryVector, indexVector3),
                        cosineSimilarity(queryVector, indexVector4) }
                );
        }
    }

    private float cosineSimilarity(float[] vectorA, float[] vectorB) {
        return 1 + VectorUtil.cosine(vectorA, vectorB);
    }

    private void validateScore(int k, float[] queryVector, float[] expectedScores) throws Exception {
        final Response responseBody = executeKNNScriptScoreRequest(testIndex, TEST_FIELD, k, SpaceType.COSINESIMIL, queryVector);
        List<Float> actualScores = parseSearchResponseScore(EntityUtils.toString(responseBody.getEntity()), TEST_FIELD);
        assertEquals(expectedScores.length, actualScores.size());
        for (int i = 0; i < expectedScores.length; i++) {
            assertEquals(expectedScores[i], actualScores.get(i), 0.01f);
        }
    }
}
