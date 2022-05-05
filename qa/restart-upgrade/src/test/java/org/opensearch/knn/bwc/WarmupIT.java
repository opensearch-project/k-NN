/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.SpaceType;

import java.util.Collections;

import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_M_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class WarmupIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int M = 50;
    private static final int EF_CONSTRUCTION = 1024;
    private static final int NUM_DOCS = 10;
    private static int QUERY_COUNT = 0;

    // Default Legacy Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKNNWarmupDefaultLegacyFieldMapping() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNWarmupOnUpgrade();
        }
    }

    // Custom Legacy Field Mapping
    // space_type : "linf", engine : "nmslib", m : 2, ef_construction : 2
    public void testKNNWarmupCustomLegacyFieldMapping() throws Exception {

        // When the cluster is in old version, create a KNN index with custom legacy field mapping settings
        // and add documents into that index
        if (isRunningAgainstOldCluster()) {
            Settings indexMappingSettings = createKNNIndexCustomLegacyFieldMappingSettings(
                SpaceType.LINF,
                KNN_ALGO_PARAM_M_MIN_VALUE,
                KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE
            );
            String indexMapping = createKnnIndexMapping(TEST_FIELD, DIMENSIONS);
            createKnnIndex(testIndex, indexMappingSettings, indexMapping);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNWarmupOnUpgrade();
        }
    }

    // Default Method Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKNNWarmupDefaultMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKNNIndexMethodFieldMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNWarmupOnUpgrade();
        }
    }

    // Custom Method Field Mapping
    // space_type : "innerproduct", engine : "faiss", m : 50, ef_construction : 1024
    public void testKNNWarmupCustomMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKNNIndexCustomMethodFieldMapping(TEST_FIELD, DIMENSIONS, SpaceType.INNER_PRODUCT, FAISS_NAME, M, EF_CONSTRUCTION)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNWarmupOnUpgrade();
        }
    }

    public void validateKNNWarmupOnUpgrade() throws Exception {
        int graphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndex));
        assertTrue(getTotalGraphsInCache() > graphCount);

        QUERY_COUNT = NUM_DOCS;
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);

        DOC_ID = NUM_DOCS;
        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

        int updatedGraphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndex));
        assertTrue(getTotalGraphsInCache() > updatedGraphCount);

        QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
        deleteKNNIndex(testIndex);
    }
}
