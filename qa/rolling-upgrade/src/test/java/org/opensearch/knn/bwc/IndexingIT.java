/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import static org.opensearch.knn.TestUtils.*;

public class IndexingIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    public void testKnnDefaultIndexSettings() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 10, K);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
                } else {
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 20, ADD_DOCS_CNT);
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 30, K);
                }
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 30, K);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 30, ADD_DOCS_CNT);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 40, K);
                forceMergeKnnIndex(testIndex);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 40, K);
                deleteKNNIndex(testIndex);
        }
    }
}
