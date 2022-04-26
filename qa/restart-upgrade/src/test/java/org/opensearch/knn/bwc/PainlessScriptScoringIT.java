/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.bwc;

public class PainlessScriptScoringIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    // KNN painless script scoring for space_type "l2"
    public void testKNNL2PainlessScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            String source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, ADD_DOCS_CNT, K);

            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);

            source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, 2 * ADD_DOCS_CNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 2 * ADD_DOCS_CNT, K);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN painless script scoring for space_type "l1"
    public void testKNNL1PainlessScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            String source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, ADD_DOCS_CNT, K);

            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);

            source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, 2 * ADD_DOCS_CNT);
            validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 2 * ADD_DOCS_CNT, K);
            deleteKNNIndex(testIndex);
        }
    }

}
