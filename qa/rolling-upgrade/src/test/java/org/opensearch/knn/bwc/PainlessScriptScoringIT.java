/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class PainlessScriptScoringIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    // KNN painless script scoring for space_type "l2"
    public void testKNNL2PainlessScriptScore() throws Exception {
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
                validateKNNL2PainlessScriptScoreOnUpgrade(totalDocsCountMixed, docIdMixed);
                break;
            case UPGRADED:
                int totalDocsCountUpgraded = 3 * NUM_DOCS;
                int docIdUpgraded = 3 * NUM_DOCS;
                validateKNNL2PainlessScriptScoreOnUpgrade(totalDocsCountUpgraded, docIdUpgraded);

                deleteKNNIndex(testIndex);
        }
    }

    // KNN painless script scoring for space_type "l1"
    public void testKNNL1PainlessScriptScore() throws Exception {
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
                validateKNNL1PainlessScriptScoreOnUpgrade(totalDocsCountMixed, docIdMixed);
                break;
            case UPGRADED:
                int totalDocsCountUpgraded = 3 * NUM_DOCS;
                int docIdUpgraded = 3 * NUM_DOCS;
                validateKNNL1PainlessScriptScoreOnUpgrade(totalDocsCountUpgraded, docIdUpgraded);

                deleteKNNIndex(testIndex);
        }
    }

    // validation steps for painless script scoring L2 after upgrading each node from old version to new version
    public void validateKNNL2PainlessScriptScoreOnUpgrade(int totalDocsCount, int docId) throws Exception {
        String source = createL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, totalDocsCount);
        validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, totalDocsCount, K);

        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);

        totalDocsCount = totalDocsCount + NUM_DOCS;
        String updatedSource = createL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, totalDocsCount);
        validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, updatedSource, totalDocsCount, K);
    }

    // validation steps for painless script scoring L1 after upgrading each node from old version to new version
    public void validateKNNL1PainlessScriptScoreOnUpgrade(int totalDocsCount, int docId) throws Exception {
        String source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, totalDocsCount);
        validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, totalDocsCount, K);

        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);

        totalDocsCount = totalDocsCount + NUM_DOCS;
        String updatedSource = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, totalDocsCount);
        validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, updatedSource, totalDocsCount, K);
    }

}
