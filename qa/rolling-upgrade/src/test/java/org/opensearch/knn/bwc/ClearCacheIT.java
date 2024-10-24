/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import java.util.Collections;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class ClearCacheIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int docId = 0;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static int queryCnt = 0;

    // Rolling Upgrade BWC Tests to validate Clear Cache API
    public void testClearCache() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                int docIdOld = 0;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                int graphCount = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndex));
                assertTrue(getTotalGraphsInCache() > graphCount);
                break;
            case UPGRADED:
                queryCnt = NUM_DOCS;
                validateClearCacheOnUpgrade(queryCnt);

                docId = NUM_DOCS;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);

                queryCnt = queryCnt + NUM_DOCS;
                validateClearCacheOnUpgrade(queryCnt);
                deleteKNNIndex(testIndex);
        }

    }

    // validation steps for Clear Cache API after upgrading all nodes from old version to new version
    public void validateClearCacheOnUpgrade(int queryCount) throws Exception {
        clearCache(Collections.singletonList(testIndex));
        assertEquals(0, getTotalGraphsInCache());
    }

}
