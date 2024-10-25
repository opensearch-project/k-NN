/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.settings.Settings;

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
            // if approximate threshold is supported, set value to 0, to build graph always
            Settings indexSettings = isApproximateThresholdSupported(getBWCVersion())
                ? buildKNNIndexSettings(0)
                : getKNNDefaultIndexSettings();
            createKnnIndex(testIndex, indexSettings, createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);
            queryCnt = NUM_DOCS;
            int graphCount = getTotalGraphsInCache();
            knnWarmup(Collections.singletonList(testIndex));
            assertTrue(getTotalGraphsInCache() > graphCount);
        } else {
            validateClearCacheOnUpgrade(queryCnt);
            deleteKNNIndex(testIndex);
        }
    }

    // validation steps for Clear Cache API after upgrading node to new version
    private void validateClearCacheOnUpgrade(int queryCount) throws Exception {
        clearCache(Collections.singletonList(testIndex));
        assertEquals(0, getTotalGraphsInCache());
    }
}
