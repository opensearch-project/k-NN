/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.client.ResponseException;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class IndexingIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    private static final String ALGO = "hnsw";

    private static final String FAISS_NAME = "faiss";
    private static final String LUCENE_NAME = "lucene";
    private static final String NMSLIB_NAME = "nmslib";

    public void testKNNDefaultIndexSettings() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
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

    /**
     * Test to verify that NMSLIB index creation is blocked in OpenSearch 3.0.0 and later,
     * while ensuring backward compatibility (BWC) for existing indexes created in OpenSearch 2.19,
     * within a rolling upgrade scenario.
     *
     * <p><b>Test Flow:</b></p>
     * <ol>
     *     <li> <b>OLD CLUSTER (OpenSearch 2.19):</b>
     *         <ul>
     *             <li>Create an index using the NMSLIB engine.</li>
     *             <li>Add sample documents.</li>
     *             <li>Flush the index to ensure it persists after upgrade.</li>
     *         </ul>
     *     </li>
     *     <li> <b>MIXED CLUSTER (Some nodes upgraded to OpenSearch 3.0.0):</b>
     *         <ul>
     *             <li>Validate that the previously created NMSLIB index is still searchable.</li>
     *             <li>Ensure documents can still be added.</li>
     *         </ul>
     *     </li>
     *     <li> <b>UPGRADED CLUSTER (All nodes upgraded to OpenSearch 3.0.0):</b>
     *         <ul>
     *             <li>Ensure the old NMSLIB index is still usable.</li>
     *             <li>Attempt to create a new index using NMSLIB → <b>Should Fail</b> with a proper error message.</li>
     *             <li>Ensure the error message matches expected behavior.</li>
     *             <li>Cleanup: Delete the original index.</li>
     *         </ul>
     *     </li>
     * </ol>
     *
     * @throws Exception if any unexpected error occurs during the test execution.
     */
    public void testBlockNMSLIBIndexCreationPost3_0_0_RollingUpgrade() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, NMSLIB_NAME));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                // Flush to ensure the index persists after upgrade
                flush(testIndex, true);
                break;

            case MIXED:
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                break;

            case UPGRADED:
                Exception ex = expectThrows(
                    ResponseException.class,
                    () -> createKnnIndex(
                        testIndex + "_new",
                        getKNNDefaultIndexSettings(),
                        createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGO, NMSLIB_NAME)
                    )
                );

                // Step 6: Cleanup - Delete the original index
                deleteKNNIndex(testIndex);
                break;
        }
    }

}
