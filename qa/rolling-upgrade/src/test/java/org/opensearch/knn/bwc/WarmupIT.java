/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import java.util.Collections;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class WarmupIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    public void testKNNWarmup() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                int docIdOld = 0;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                break;
            case MIXED:
                int totalDocsCountMixed;
                int docIdMixed;
                if (isFirstMixedRound()) {
                    docIdMixed = NUM_DOCS;
                    totalDocsCountMixed = 2 * NUM_DOCS;
                } else {
                    docIdMixed = 2 * NUM_DOCS;
                    totalDocsCountMixed = 3 * NUM_DOCS;
                }
                validateKNNWarmupOnUpgrade(totalDocsCountMixed, docIdMixed);
                break;
            case UPGRADED:
                int docIdUpgraded = 3 * NUM_DOCS;
                int totalDocsCountUpgraded = 4 * NUM_DOCS;
                validateKNNWarmupOnUpgrade(totalDocsCountUpgraded, docIdUpgraded);

                deleteKNNIndex(testIndex);
        }

    }

    // validation steps for KNN Warmup after upgrading each node from old version to new version
    public void validateKNNWarmupOnUpgrade(int totalDocsCount, int docId) throws Exception {
        int graphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndex));
        assertTrue(getTotalGraphsInCache() > graphCount);

        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);

        int updatedGraphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndex));
        assertTrue(getTotalGraphsInCache() > updatedGraphCount);

        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, totalDocsCount, K);
    }

}
