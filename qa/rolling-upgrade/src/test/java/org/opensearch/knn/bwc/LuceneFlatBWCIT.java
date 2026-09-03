/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;

public class LuceneFlatBWCIT extends AbstractRollingUpgradeTestCase {

    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSION = 8;
    private static final int NUM_DOCS = 10;
    private static final int K = 5;

    /**
     * Rolling upgrade variant of the flat + explicit engine BWC check. The old cluster creates a
     * flat mapping with engine=lucene explicitly set. As nodes roll to the new version that
     * enforces engine-agnostic flat, the mapping (which still carries engine=lucene) must
     * continue to load, be searchable in mixed and upgraded phases.
     */
    public void testRollingUpgrade_flatWithExplicitLuceneEngine() throws Exception {
        // Only meaningful when the old cluster still accepts engine on method=flat.
        // Once the old cluster is on the engine-agnostic gate version or later, it rejects the
        // mapping itself and there is no pre-gate → post-gate transition to exercise.
        if (isFlatMethodEnginePermittedOnOldCluster(getBWCVersion()) == false) {
            logger.info("Skipping test — flat + explicit engine not accepted by old cluster in version: {}", getBWCVersion());
            return;
        }
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        switch (getClusterType()) {
            case OLD:
                XContentBuilder mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(TEST_FIELD)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .startObject(KNN_METHOD)
                    .field(NAME, METHOD_FLAT)
                    .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                    .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping.toString());
                addKNNDocs(testIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                flush(testIndex, true);
                break;

            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);
                break;

            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS, NUM_DOCS);
                forceMergeKnnIndex(testIndex);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSION, 2 * NUM_DOCS, K);
                deleteKNNIndex(testIndex);
        }
    }
}
