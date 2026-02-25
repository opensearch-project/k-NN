/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

public class PainlessScriptScoringIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static int QUERY_COUNT = 0;

    // KNN painless script scoring for space_type "l2"
    public void testKNNL2PainlessScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            DOC_ID = NUM_DOCS;
            QUERY_COUNT = NUM_DOCS;
            String source = createL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);

            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            source = createL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN painless script scoring for space_type "l1"
    public void testKNNL1PainlessScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, createKNNDefaultScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            DOC_ID = NUM_DOCS;
            QUERY_COUNT = NUM_DOCS;
            String source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);

            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            deleteKNNIndex(testIndex);
        }
    }

    public void testNonKNNIndex_withMethodParams_withFaissEngine() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                createKNNDefaultScriptScoreSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, "hnsw", KNNEngine.FAISS.getName(), SpaceType.DEFAULT.getValue(), false)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            DOC_ID = NUM_DOCS;
            QUERY_COUNT = NUM_DOCS;
            String source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            deleteKNNIndex(testIndex);
        }
    }

    public void testNonKNNIndex_withMethodParams_withNmslibEngine() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                createKNNDefaultScriptScoreSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, "hnsw", KNNEngine.NMSLIB.getName(), SpaceType.DEFAULT.getValue(), false)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            DOC_ID = NUM_DOCS;
            QUERY_COUNT = NUM_DOCS;
            String source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            deleteKNNIndex(testIndex);
        }
    }

    public void testNonKNNIndex_withMethodParams_withLuceneEngine() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                createKNNDefaultScriptScoreSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, "hnsw", KNNEngine.LUCENE.getName(), SpaceType.DEFAULT.getValue(), false)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            DOC_ID = NUM_DOCS;
            QUERY_COUNT = NUM_DOCS;
            String source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
            source = createL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
            deleteKNNIndex(testIndex);
        }
    }
}
