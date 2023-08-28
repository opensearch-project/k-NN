/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import java.util.Collections;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class ClearCacheIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int docId = 0;
    private static final int NUM_DOCS = 10;
    private static int queryCnt = 0;
    private static final int K = 5;

    // Restart Upgrade BWC Tests to validate Clear Cache API
    public void testClearCache() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);
        } else {
            queryCnt = NUM_DOCS;
            validateClearCacheOnUpgrade(queryCnt);

            docId = NUM_DOCS;
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);

            queryCnt = queryCnt + NUM_DOCS;
            validateClearCacheOnUpgrade(queryCnt);
            deleteKNNIndex(testIndex);
        }
    }

    // validation steps for Clear Cache API after upgrading node to new version
    private void validateClearCacheOnUpgrade(int queryCount) throws Exception {
        int graphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndex));
        assertTrue(getTotalGraphsInCache() > graphCount);
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, queryCount, K);

        clearCache(Collections.singletonList(testIndex));
        assertEquals(0, getTotalGraphsInCache());
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, queryCount, K);
    }
}
