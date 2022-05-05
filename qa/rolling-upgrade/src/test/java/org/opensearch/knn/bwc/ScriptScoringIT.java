/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.knn.index.SpaceType;

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

}
