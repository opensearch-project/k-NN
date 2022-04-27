/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;

import java.util.List;
import java.util.Set;

import static org.opensearch.knn.TestUtils.KNN_BWC_PREFIX;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY;

public class RecallIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_INDEX_RECALL_OLD = KNN_BWC_PREFIX + "test-index-recall-old";
    private static final String TEST_FIELD = "test-field";
    private static final double DELTA_VALUE = 0.2;
    private static final int DIMENSIONS = 50;
    private static final int DIMENSIONS_RECALL_OLD = 1;
    private static final String DOC_ID = "1";
    private static final int INDEX_THREAD_QTY = 2;
    private static final int K = 10;
    private static final int NUM_DOCS = 500;
    private static final int QUERY_COUNT = 100;

    public void testKNNRecall() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, true);

            double recallVal = getKNNBWCRecallValue(testIndex, TEST_FIELD, NUM_DOCS, DIMENSIONS, QUERY_COUNT, K, true, SpaceType.L2);
            createKnnIndex(TEST_INDEX_RECALL_OLD, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS_RECALL_OLD));
            addKnnDoc(TEST_INDEX_RECALL_OLD, DOC_ID, TEST_FIELD, new Float[] { (float) recallVal });

        } else {
            double recallValUpgraded = getKNNBWCRecallValue(
                testIndex,
                TEST_FIELD,
                NUM_DOCS,
                DIMENSIONS,
                QUERY_COUNT,
                K,
                true,
                SpaceType.L2
            );
            float[][] expRecallValue = getIndexVectorsFromIndex(TEST_INDEX_RECALL_OLD, TEST_FIELD, 1, DIMENSIONS_RECALL_OLD);
            assertEquals(expRecallValue[0][0], recallValUpgraded, DELTA_VALUE);

            deleteKNNIndex(testIndex);
            deleteKNNIndex(TEST_INDEX_RECALL_OLD);
        }
    }

    private double getKNNBWCRecallValue(
        String testIndex,
        String testField,
        int docCount,
        int dimensions,
        int queryCount,
        int k,
        boolean isStandard,
        SpaceType spaceType
    ) throws Exception {
        float[][] indexVectors = getIndexVectorsFromIndex(testIndex, testField, docCount, dimensions);
        float[][] queryVectors = TestUtils.getQueryVectors(queryCount, dimensions, docCount, isStandard);
        List<Set<String>> groundTruthValues = TestUtils.computeGroundTruthValues(indexVectors, queryVectors, spaceType, k);
        List<List<String>> searchResults = bulkSearch(testIndex, testField, queryVectors, k);
        return TestUtils.calculateRecallValue(searchResults, groundTruthValues, k);
    }

    private void addDocs(String testIndex, String testField, int dimensions, int docCount, boolean isStandard) throws Exception {

        // Updating index thread qty to 2 to speed up data ingestion
        updateClusterSettings(KNN_ALGO_PARAM_INDEX_THREAD_QTY, INDEX_THREAD_QTY);

        bulkAddKnnDocs(testIndex, testField, TestUtils.getIndexVectors(docCount, dimensions, isStandard), docCount);
    }
}
