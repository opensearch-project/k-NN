/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;

import java.util.Collections;
import java.util.List;

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
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                break;
            case MIXED:
                int graphCount = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndex));
                assertTrue(getTotalGraphsInCache() > graphCount);
                clearCache(List.of(testIndex));
                break;
            case UPGRADED:
                updateIndexSettings(testIndex, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                int updatedGraphCount = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndex));
                assertTrue(getTotalGraphsInCache() > updatedGraphCount);
                deleteKNNIndex(testIndex);
        }
    }

}
