/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class IndexingIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    private static final String ALGO = "hnsw";

    private static final String FAISS_NAME = "faiss";
    private static final String LUCENE_NAME = "lucene";

    public void testKNNDefaultIndexSettings() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                int docIdOld = 0;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                break;
            case MIXED:
                int totalDocsCountMixed;
                int docIdMixed;
                if (isFirstMixedRound()) {
                    totalDocsCountMixed = NUM_DOCS;
                    docIdMixed = NUM_DOCS;
                } else {
                    totalDocsCountMixed = 2 * NUM_DOCS;
                    docIdMixed = 2 * NUM_DOCS;
                }
                validateKNNIndexingOnUpgrade(totalDocsCountMixed, docIdMixed);
                break;
            case UPGRADED:
                int totalDocsCountUpgraded = 3 * NUM_DOCS;
                int docIdUpgraded = 3 * NUM_DOCS;
                validateKNNIndexingOnUpgrade(totalDocsCountUpgraded, docIdUpgraded);

                int updatedDocIdUpgraded = docIdUpgraded + NUM_DOCS;
                forceMergeKnnIndex(testIndex);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, updatedDocIdUpgraded, K);

                deleteKNNIndex(testIndex);
        }
    }

    public void testKNNIndexCreation_withLegacyMapper() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        final String firstMixRoundIndex = testIndex + "first-mix-round";
        final String otherMixRoundIndex = testIndex + "other-mix-round";
        final String upgradedIndex = testIndex + "upgraded";
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                int docIdOld = 0;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    docIdOld = 0;
                    createKnnIndex(firstMixRoundIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                    addKNNDocs(firstMixRoundIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                } else {
                    docIdOld = 0;
                    createKnnIndex(otherMixRoundIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                    addKNNDocs(otherMixRoundIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                }
                break;
            case UPGRADED:
                docIdOld = 0;
                createKnnIndex(upgradedIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(upgradedIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);

                deleteKNNIndex(testIndex);
                deleteKNNIndex(firstMixRoundIndex);
                deleteKNNIndex(otherMixRoundIndex);
                deleteKNNIndex(upgradedIndex);
        }
    }

    public void testKNNIndexCreation_withMethodMapper() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        final String firstMixRoundIndex = testIndex + "first-mix-round";
        final String otherMixRoundIndex = testIndex + "other-mix-round";
        final String upgradedIndex = testIndex + "upgraded";
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, FAISS_NAME));
                int docIdOld = 0;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    docIdOld = 0;
                    createKnnIndex(
                        firstMixRoundIndex,
                        getKNNDefaultIndexSettings(),
                        createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, FAISS_NAME)
                    );
                    addKNNDocs(firstMixRoundIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                } else {
                    docIdOld = 0;
                    createKnnIndex(
                        otherMixRoundIndex,
                        getKNNDefaultIndexSettings(),
                        createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, FAISS_NAME)
                    );
                    addKNNDocs(otherMixRoundIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                }
                break;
            case UPGRADED:
                docIdOld = 0;
                createKnnIndex(
                    upgradedIndex,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, FAISS_NAME)
                );
                addKNNDocs(upgradedIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);

                deleteKNNIndex(testIndex);
                deleteKNNIndex(firstMixRoundIndex);
                deleteKNNIndex(otherMixRoundIndex);
                deleteKNNIndex(upgradedIndex);
        }
    }

    public void testKNNLuceneIndexCreation_withMethodMapper() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        final String firstMixRoundIndex = testIndex + "first-mix-round";
        final String otherMixRoundIndex = testIndex + "other-mix-round";
        final String upgradedIndex = testIndex + "upgraded";
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, LUCENE_NAME));
                int docIdOld = 0;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    docIdOld = 0;
                    createKnnIndex(
                        firstMixRoundIndex,
                        getKNNDefaultIndexSettings(),
                        createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, LUCENE_NAME)
                    );
                    addKNNDocs(firstMixRoundIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                } else {
                    docIdOld = 0;
                    createKnnIndex(
                        otherMixRoundIndex,
                        getKNNDefaultIndexSettings(),
                        createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, LUCENE_NAME)
                    );
                    addKNNDocs(otherMixRoundIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);
                }
                break;
            case UPGRADED:
                docIdOld = 0;
                createKnnIndex(
                    upgradedIndex,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, LUCENE_NAME)
                );
                addKNNDocs(upgradedIndex, TEST_FIELD, DIMENSIONS, docIdOld, NUM_DOCS);

                deleteKNNIndex(testIndex);
                deleteKNNIndex(firstMixRoundIndex);
                deleteKNNIndex(otherMixRoundIndex);
                deleteKNNIndex(upgradedIndex);
        }
    }

    // validation steps for indexing after upgrading each node from old version to new version
    public void validateKNNIndexingOnUpgrade(int totalDocsCount, int docId) throws Exception {
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, totalDocsCount, K);
        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, docId, NUM_DOCS);

        totalDocsCount = totalDocsCount + NUM_DOCS;
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, totalDocsCount, K);
    }
}
