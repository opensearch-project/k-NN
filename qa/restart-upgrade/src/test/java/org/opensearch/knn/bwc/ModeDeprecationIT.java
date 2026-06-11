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

/**
 * Restart-upgrade BwC tests for mode parameter deprecation (V_3_7_0+).
 *
 * <p>Covers the following backward-compatibility scenarios:
 * <ol>
 *   <li>An index created with {@code mode=on_disk} on the old cluster must survive a full
 *       cluster restart-upgrade and remain searchable on the new cluster.</li>
 *   <li>An index created with {@code mode=in_memory} on the old cluster must survive
 *       restart-upgrade and remain searchable.</li>
 *   <li>An index created with {@code mode=on_disk} only (no explicit compression) on the old
 *       cluster must survive restart-upgrade; the upgraded cluster re-derives compression from
 *       the stored mode value and produces the same result.</li>
 *   <li>On the upgraded cluster, an index created with only {@code compression_level} (no
 *       explicit {@code mode}) must resolve correctly via the new derivation logic.</li>
 *   <li>On the upgraded cluster, providing both {@code compression_level} and {@code mode}
 *       must still work (deprecated but honoured) and produce a deprecation warning in logs.</li>
 * </ol>
 *
 * <p>Tests that require the old cluster to support mode/compression (>= 2.17.0) are guarded
 * by {@link #isModeSupported(java.util.Optional)}.
 */
public class ModeDeprecationIT extends AbstractRestartUpgradeTestCase {

    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSION = 16;
    private static final int K = 5;
    private static final int NUM_DOCS = 100;

    // -------------------------------------------------------------------------
    // Scenario 1: mode=on_disk + compression_level=32x created on old cluster
    // -------------------------------------------------------------------------

    /**
     * Creates an index with explicit {@code mode=on_disk, compression_level=32x} on the old
     * cluster. After restart-upgrade the index must still be searchable and accept new docs.
     * The mapping stores the user-provided {@code mode}; the upgraded cluster must honour it
     * without re-deriving or rejecting it (BwC guarantee).
     */
    public void testOnDiskModeIndex_onOldClusterThenUpgraded_thenSucceed() throws Exception {
        if (isModeSupported(getBWCVersion()) == false) {
            logger.info("Skipping: mode parameter not supported in BWC version {}", getBWCVersion());
            return;
        }
        String indexName = testIndex + "-ondisk-32x";
        if (isRunningAgainstOldCluster()) {
            createOnDiskIndex(indexName, "32x");
            addKNNDocs(indexName, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
        } else {
            // Upgraded cluster: existing index must be searchable
            validateKNNSearch(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
            // New docs must be indexable
            addKNNDocs(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(indexName);
            validateKNNSearch(indexName, TEST_FIELD, DIMENSION, 2 * NUM_DOCS, K);
            deleteKNNIndex(indexName);
        }
    }

    // -------------------------------------------------------------------------
    // Scenario 2: mode=in_memory + compression_level=1x created on old cluster
    // -------------------------------------------------------------------------

    /**
     * Creates an index with explicit {@code mode=in_memory, compression_level=1x} on the old
     * cluster. After restart-upgrade the index must still be searchable.
     */
    public void testInMemoryModeIndex_onOldClusterThenUpgraded_thenSucceed() throws Exception {
        if (isModeSupported(getBWCVersion()) == false) {
            logger.info("Skipping: mode parameter not supported in BWC version {}", getBWCVersion());
            return;
        }
        String indexName = testIndex + "-inmemory-1x";
        if (isRunningAgainstOldCluster()) {
            createInMemoryIndex(indexName, "1x");
            addKNNDocs(indexName, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
        } else {
            validateKNNSearch(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
            addKNNDocs(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(indexName);
            validateKNNSearch(indexName, TEST_FIELD, DIMENSION, 2 * NUM_DOCS, K);
            deleteKNNIndex(indexName);
        }
    }

    // -------------------------------------------------------------------------
    // Scenario 3: mode=on_disk only (no compression) created on old cluster
    // -------------------------------------------------------------------------

    /**
     * Creates an index with only {@code mode=on_disk} (no explicit compression) on the old
     * cluster. The old cluster resolves this to {@code compression_level=32x} internally.
     * After restart-upgrade the upgraded cluster must re-derive the same configuration from
     * the stored mapping and the index must remain searchable.
     */
    public void testOnDiskModeOnlyIndex_onOldClusterThenUpgraded_thenSucceed() throws Exception {
        if (isModeSupported(getBWCVersion()) == false) {
            logger.info("Skipping: mode parameter not supported in BWC version {}", getBWCVersion());
            return;
        }
        String indexName = testIndex + "-ondisk-only";
        if (isRunningAgainstOldCluster()) {
            createModeOnlyIndex(indexName, "on_disk");
            addKNNDocs(indexName, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
        } else {
            validateKNNSearch(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
            addKNNDocs(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(indexName);
            validateKNNSearch(indexName, TEST_FIELD, DIMENSION, 2 * NUM_DOCS, K);
            deleteKNNIndex(indexName);
        }
    }

    // -------------------------------------------------------------------------
    // Scenario 4: compression_level only (no mode) on upgraded cluster
    // -------------------------------------------------------------------------

    /**
     * On the upgraded cluster only, creates an index with {@code compression_level=32x} and
     * no explicit {@code mode}. The resolver must derive {@code mode=on_disk} automatically
     * and the index must be searchable.
     */
    public void testCompressionOnly32x_onUpgradedCluster_thenSucceed() throws Exception {
        if (isRunningAgainstOldCluster()) {
            return;
        }
        String indexName = testIndex + "-compression-32x";
        createCompressionOnlyIndex(indexName, "32x");
        addKNNDocs(indexName, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
        forceMergeKnnIndex(indexName);
        validateKNNSearch(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
        deleteKNNIndex(indexName);
    }

    /**
     * On the upgraded cluster only, creates an index with {@code compression_level=2x} and
     * no explicit {@code mode}. The resolver must derive {@code mode=in_memory} automatically.
     */
    public void testCompressionOnly2x_onUpgradedCluster_thenSucceed() throws Exception {
        if (isRunningAgainstOldCluster()) {
            return;
        }
        String indexName = testIndex + "-compression-2x";
        createCompressionOnlyIndex(indexName, "2x");
        addKNNDocs(indexName, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
        forceMergeKnnIndex(indexName);
        validateKNNSearch(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
        deleteKNNIndex(indexName);
    }

    /**
     * On the upgraded cluster only, creates an index with {@code compression_level=1x} and
     * no explicit {@code mode}. The resolver must derive {@code mode=in_memory} automatically.
     */
    public void testCompressionOnly1x_onUpgradedCluster_thenSucceed() throws Exception {
        if (isRunningAgainstOldCluster()) {
            return;
        }
        String indexName = testIndex + "-compression-1x";
        createCompressionOnlyIndex(indexName, "1x");
        addKNNDocs(indexName, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
        forceMergeKnnIndex(indexName);
        validateKNNSearch(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
        deleteKNNIndex(indexName);
    }

    // -------------------------------------------------------------------------
    // Scenario 5: both compression_level and mode on upgraded cluster (deprecated but honoured)
    // -------------------------------------------------------------------------

    /**
     * On the upgraded cluster, providing both {@code compression_level=32x} and
     * {@code mode=on_disk} must still succeed (deprecated but honoured for BwC).
     * A deprecation warning is expected in the server logs but the index must be functional.
     */
    public void testExplicitModeAndCompression_onUpgradedCluster_thenSucceed() throws Exception {
        if (isRunningAgainstOldCluster()) {
            return;
        }
        String indexName = testIndex + "-explicit-both";
        createOnDiskIndex(indexName, "32x");
        addKNNDocs(indexName, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
        forceMergeKnnIndex(indexName);
        validateKNNSearch(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
        deleteKNNIndex(indexName);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private void createOnDiskIndex(String indexName, String compressionLevel) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "on_disk")
            .field("compression_level", compressionLevel)
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
    }

    private void createInMemoryIndex(String indexName, String compressionLevel) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "in_memory")
            .field("compression_level", compressionLevel)
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
    }

    private void createModeOnlyIndex(String indexName, String mode) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", mode)
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
