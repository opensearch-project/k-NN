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

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

/**
 * Rolling-upgrade BwC tests for mode parameter deprecation (V_3_7_0+).
 *
 * <p>Covers three backward-compatibility scenarios:
 * <ol>
 *   <li>An index created with {@code mode=on_disk} on the old cluster must survive rolling
 *       upgrade and remain searchable on mixed and fully-upgraded clusters.</li>
 *   <li>An index created with {@code mode=in_memory} on the old cluster must survive rolling
 *       upgrade and remain searchable.</li>
 *   <li>On the fully-upgraded cluster, an index created with only {@code compression_level}
 *       (no explicit {@code mode}) must resolve correctly via the new derivation logic and
 *       be searchable.</li>
 * </ol>
 *
 * <p>These tests do NOT run on old clusters that pre-date mode/compression support (< 2.17.0).
 */
public class ModeDeprecationIT extends AbstractRollingUpgradeTestCase {

    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSION = 16;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    // -------------------------------------------------------------------------
    // Scenario 1: mode=on_disk index created on old cluster survives upgrade
    // -------------------------------------------------------------------------

    /**
     * Creates an index with {@code mode=on_disk, compression_level=32x} on the old cluster,
     * continues indexing through mixed rounds, and verifies search on the upgraded cluster.
     * The mapping stores the explicit {@code mode} value; the upgraded cluster must honour it
     * without re-deriving or rejecting it.
     */
    public void testOnDiskModeIndex_rollingUpgrade_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isModeSupported(getBWCVersion()) == false) {
            return;
        }
        String onDiskIndex = testIndex + "-ondisk";
        switch (getClusterType()) {
            case OLD:
                createOnDiskIndex(onDiskIndex);
                addKNNDocs(onDiskIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                break;
            case MIXED:
                int totalDocs = isFirstMixedRound() ? 2 * NUM_DOCS : 3 * NUM_DOCS;
                int docId = isFirstMixedRound() ? NUM_DOCS : 2 * NUM_DOCS;
                addKNNDocs(onDiskIndex, TEST_FIELD, DIMENSION, docId, NUM_DOCS);
                validateKNNSearch(onDiskIndex, TEST_FIELD, DIMENSION, totalDocs, K);
                break;
            case UPGRADED:
                addKNNDocs(onDiskIndex, TEST_FIELD, DIMENSION, 3 * NUM_DOCS, NUM_DOCS);
                forceMergeKnnIndex(onDiskIndex);
                validateKNNSearch(onDiskIndex, TEST_FIELD, DIMENSION, 4 * NUM_DOCS, K);
                deleteKNNIndex(onDiskIndex);
        }
    }

    // -------------------------------------------------------------------------
    // Scenario 2: mode=in_memory index created on old cluster survives upgrade
    // -------------------------------------------------------------------------

    /**
     * Creates an index with {@code mode=in_memory, compression_level=1x} on the old cluster
     * and verifies it remains searchable through rolling upgrade.
     */
    public void testInMemoryModeIndex_rollingUpgrade_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isModeSupported(getBWCVersion()) == false) {
            return;
        }
        String inMemoryIndex = testIndex + "-inmemory";
        switch (getClusterType()) {
            case OLD:
                createInMemoryIndex(inMemoryIndex);
                addKNNDocs(inMemoryIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                break;
            case MIXED:
                int totalDocs = isFirstMixedRound() ? 2 * NUM_DOCS : 3 * NUM_DOCS;
                int docId = isFirstMixedRound() ? NUM_DOCS : 2 * NUM_DOCS;
                addKNNDocs(inMemoryIndex, TEST_FIELD, DIMENSION, docId, NUM_DOCS);
                validateKNNSearch(inMemoryIndex, TEST_FIELD, DIMENSION, totalDocs, K);
                break;
            case UPGRADED:
                addKNNDocs(inMemoryIndex, TEST_FIELD, DIMENSION, 3 * NUM_DOCS, NUM_DOCS);
                forceMergeKnnIndex(inMemoryIndex);
                validateKNNSearch(inMemoryIndex, TEST_FIELD, DIMENSION, 4 * NUM_DOCS, K);
                deleteKNNIndex(inMemoryIndex);
        }
    }

    // -------------------------------------------------------------------------
    // Scenario 3: compression_level only (no mode) on upgraded cluster
    // -------------------------------------------------------------------------

    /**
     * On the fully-upgraded cluster, creates an index with only {@code compression_level=32x}
     * (no explicit {@code mode}). The resolver must derive {@code mode=on_disk} automatically
     * and the index must be searchable.
     */
    public void testCompressionOnlyIndex_onUpgradedCluster_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
            case MIXED:
                break;
            case UPGRADED:
                String compressionOnlyIndex = testIndex + "-compressiononly";
                createCompressionOnlyIndex(compressionOnlyIndex, "32x");
                addKNNDocs(compressionOnlyIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                forceMergeKnnIndex(compressionOnlyIndex);
                validateKNNSearch(compressionOnlyIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);
                deleteKNNIndex(compressionOnlyIndex);
        }
    }

    /**
     * On the fully-upgraded cluster, creates an index with only {@code compression_level=2x}
     * (no explicit {@code mode}). The resolver must derive {@code mode=in_memory} automatically.
     */
    public void testCompressionOnly2x_onUpgradedCluster_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
            case MIXED:
                break;
            case UPGRADED:
                String compressionOnly2xIndex = testIndex + "-compressiononly2x";
                createCompressionOnlyIndex(compressionOnly2xIndex, "2x");
                addKNNDocs(compressionOnly2xIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                forceMergeKnnIndex(compressionOnly2xIndex);
                validateKNNSearch(compressionOnly2xIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);
                deleteKNNIndex(compressionOnly2xIndex);
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private void createOnDiskIndex(String indexName) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "on_disk")
            .field("compression_level", "32x")
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
    }

    private void createInMemoryIndex(String indexName) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "in_memory")
            .field("compression_level", "1x")
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
    }

    private void createCompressionOnlyIndex(String indexName, String compressionLevel) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("compression_level", compressionLevel)
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
    }

    /**
     * Returns true if the old cluster version supports the mode/compression parameters
     * (introduced in 2.17.0).
     */
    private boolean isModeSupported(java.util.Optional<String> bwcVersion) {
        if (bwcVersion.isEmpty()) {
            return false;
        }
        String versionString = bwcVersion.get().replace("-SNAPSHOT", "");
        return Version.fromString(versionString).onOrAfter(Version.V_2_17_0);
    }
}
