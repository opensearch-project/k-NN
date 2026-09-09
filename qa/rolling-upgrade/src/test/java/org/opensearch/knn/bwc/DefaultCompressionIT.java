/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;

public class DefaultCompressionIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 64;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    @SuppressWarnings("unchecked")
    public void testRollingUpgrade_defaultCompression() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        final String explicitX32Index = testIndex + "-explicit-x32";
        final boolean compressionSupported = isCompressionSupported(getBWCVersion());

        switch (getClusterType()) {
            case OLD:
                if (compressionSupported) {
                    String explicitMapping = XContentFactory.jsonBuilder()
                        .startObject()
                        .startObject(PROPERTIES)
                        .startObject(TEST_FIELD)
                        .field(VECTOR_TYPE, KNN_VECTOR)
                        .field(DIMENSION, DIMENSIONS)
                        .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                        .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                        .endObject()
                        .endObject()
                        .endObject()
                        .toString();
                    createKnnIndex(explicitX32Index, getKNNDefaultIndexSettings(), explicitMapping);
                    addKNNDocs(explicitX32Index, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                    flush(explicitX32Index, true);
                }

                createKnnIndex(
                    testIndex,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, FAISS_NAME)
                );
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;

            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                Map<String, Object> defaultMappings = getIndexMappingAsMap(testIndex);
                Map<String, Object> defaultProperties = (Map<String, Object>) defaultMappings.get(PROPERTIES);
                assertNotNull("Properties should not be null", defaultProperties);
                Map<String, Object> defaultFieldProps = (Map<String, Object>) defaultProperties.get(TEST_FIELD);
                assertNotNull("Field properties should not be null", defaultFieldProps);
                assertNull(defaultFieldProps.get(COMPRESSION_LEVEL_PARAMETER));

                if (compressionSupported) {
                    validateKNNSearch(explicitX32Index, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                    Map<String, Object> x32Mappings = getIndexMappingAsMap(explicitX32Index);
                    Map<String, Object> x32Properties = (Map<String, Object>) x32Mappings.get(PROPERTIES);
                    assertNotNull("x32 properties should not be null", x32Properties);
                    Map<String, Object> x32FieldProps = (Map<String, Object>) x32Properties.get(TEST_FIELD);
                    assertNotNull("x32 field properties should not be null", x32FieldProps);
                    assertEquals(CompressionLevel.x32.getName(), x32FieldProps.get(COMPRESSION_LEVEL_PARAMETER));
                }
                break;

            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                Map<String, Object> upgradedDefaultMappings = getIndexMappingAsMap(testIndex);
                Map<String, Object> upgradedDefaultProperties = (Map<String, Object>) upgradedDefaultMappings.get(PROPERTIES);
                assertNotNull("Properties should not be null after upgrade", upgradedDefaultProperties);
                Map<String, Object> upgradedDefaultFieldProps = (Map<String, Object>) upgradedDefaultProperties.get(TEST_FIELD);
                assertNotNull("Field properties should not be null after upgrade", upgradedDefaultFieldProps);
                assertNull(upgradedDefaultFieldProps.get(COMPRESSION_LEVEL_PARAMETER));

                if (compressionSupported) {
                    validateKNNSearch(explicitX32Index, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                    Map<String, Object> upgradedX32Mappings = getIndexMappingAsMap(explicitX32Index);
                    Map<String, Object> upgradedX32Properties = (Map<String, Object>) upgradedX32Mappings.get(PROPERTIES);
                    assertNotNull("x32 properties should not be null after upgrade", upgradedX32Properties);
                    Map<String, Object> upgradedX32FieldProps = (Map<String, Object>) upgradedX32Properties.get(TEST_FIELD);
                    assertNotNull("x32 field properties should not be null after upgrade", upgradedX32FieldProps);
                    assertEquals(CompressionLevel.x32.getName(), upgradedX32FieldProps.get(COMPRESSION_LEVEL_PARAMETER));
                    deleteKNNIndex(explicitX32Index);
                }

                // TODO: [DEFAULT_FLIP] After Step 4, add test that post-upgrade NEW index creation (implicit) resolves to SQ 1-bit

                deleteKNNIndex(testIndex);
        }
    }

    public void testRollingUpgrade_mixedClusterOperations() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        final String fp32Index = testIndex + "-fp32-forcemerge";

        switch (getClusterType()) {
            case OLD:
                createKnnIndex(
                    fp32Index,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, FAISS_NAME)
                );
                addKNNDocs(fp32Index, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(fp32Index, true);
                break;

            case MIXED:
                if (isFirstMixedRound()) {
                    validateKNNSearch(fp32Index, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                    addKNNDocs(fp32Index, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                    flush(fp32Index, true);
                } else {
                    int mixedTotal = 2 * NUM_DOCS;
                    validateKNNSearch(fp32Index, TEST_FIELD, DIMENSIONS, mixedTotal, K);
                }
                break;

            case UPGRADED:
                int totalDocs = 2 * NUM_DOCS;
                validateKNNSearch(fp32Index, TEST_FIELD, DIMENSIONS, totalDocs, K);
                forceMergeKnnIndex(fp32Index);
                validateKNNSearch(fp32Index, TEST_FIELD, DIMENSIONS, totalDocs, K);
                deleteKNNIndex(fp32Index);
        }
    }

    public void testRollingUpgrade_mixedSegments() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        final String fp32Index = testIndex + "-fp32-mixed-segments";

        switch (getClusterType()) {
            case OLD:
                createKnnIndex(
                    fp32Index,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, FAISS_NAME)
                );
                addKNNDocs(fp32Index, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(fp32Index, true);
                break;

            case MIXED:
                validateKNNSearch(fp32Index, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                break;

            case UPGRADED:
                addKNNDocs(fp32Index, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                flush(fp32Index, true);

                int totalDocs = 2 * NUM_DOCS;
                validateKNNSearch(fp32Index, TEST_FIELD, DIMENSIONS, totalDocs, K);

                forceMergeKnnIndex(fp32Index);
                validateKNNSearch(fp32Index, TEST_FIELD, DIMENSIONS, totalDocs, K);

                deleteKNNIndex(fp32Index);
        }
    }

    @SuppressWarnings("unchecked")
    public void testRollingUpgrade_explicitCompression() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isCompressionSupported(getBWCVersion()) == false) {
            return;
        }
        final String preUpgradeIndex = testIndex + "-pre-upgrade-x32";
        final String postUpgradeIndex = testIndex + "-post-upgrade-x32";

        switch (getClusterType()) {
            case OLD:
                String x32Mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(preUpgradeIndex, getKNNDefaultIndexSettings(), x32Mapping);
                addKNNDocs(preUpgradeIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(preUpgradeIndex, true);
                break;

            case MIXED:
                validateKNNSearch(preUpgradeIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                break;

            case UPGRADED:
                validateKNNSearch(preUpgradeIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                Map<String, Object> preUpgradeMappings = getIndexMappingAsMap(preUpgradeIndex);
                Map<String, Object> preUpgradeProperties = (Map<String, Object>) preUpgradeMappings.get(PROPERTIES);
                assertNotNull("Pre-upgrade properties should not be null", preUpgradeProperties);
                Map<String, Object> preUpgradeFieldProps = (Map<String, Object>) preUpgradeProperties.get(TEST_FIELD);
                assertNotNull("Pre-upgrade field properties should not be null", preUpgradeFieldProps);
                assertEquals(CompressionLevel.x32.getName(), preUpgradeFieldProps.get(COMPRESSION_LEVEL_PARAMETER));
                assertEquals(Mode.ON_DISK.getName(), preUpgradeFieldProps.get(MODE_PARAMETER));

                String postUpgradeMapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(postUpgradeIndex, getKNNDefaultIndexSettings(), postUpgradeMapping);
                addKNNDocs(postUpgradeIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                validateKNNSearch(postUpgradeIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                Map<String, Object> postUpgradeMappings = getIndexMappingAsMap(postUpgradeIndex);
                Map<String, Object> postUpgradeProperties = (Map<String, Object>) postUpgradeMappings.get(PROPERTIES);
                assertNotNull("Post-upgrade properties should not be null", postUpgradeProperties);
                Map<String, Object> postUpgradeFieldProps = (Map<String, Object>) postUpgradeProperties.get(TEST_FIELD);
                assertNotNull("Post-upgrade field properties should not be null", postUpgradeFieldProps);
                assertEquals(CompressionLevel.x32.getName(), postUpgradeFieldProps.get(COMPRESSION_LEVEL_PARAMETER));
                assertEquals(Mode.ON_DISK.getName(), postUpgradeFieldProps.get(MODE_PARAMETER));

                deleteKNNIndex(preUpgradeIndex);
                deleteKNNIndex(postUpgradeIndex);
        }
    }

    /**
     * BWC coverage for the Faiss x8 / x16 default-encoder switch. Before {@link Version#V_3_9_0}
     * the Faiss resolver auto-picked BQ (QFrame) 4-bit (x8) / 2-bit (x16); from that version
     * onward it picks Faiss SQ 4-bit / 2-bit. The mapping API does not surface the auto-resolved
     * encoder in the stored JSON (only user-submitted fields round-trip), so this test doesn't
     * assert on encoder identity — it verifies pre-gate BQ segments continue to load, search,
     * accept new writes, and survive a force-merge across mixed and upgraded phases.
     */
    public void testRollingUpgrade_faissX16DefaultSwitch_persistsBQ() throws Exception {
        runFaissDefaultSwitchTest(CompressionLevel.x16, "-x16");
    }

    public void testRollingUpgrade_faissX8DefaultSwitch_persistsBQ() throws Exception {
        runFaissDefaultSwitchTest(CompressionLevel.x8, "-x8");
    }

    @SuppressWarnings("unchecked")
    private void runFaissDefaultSwitchTest(CompressionLevel compressionLevel, String indexSuffix) throws Exception {
        // Only meaningful when the old cluster is in [V_2_17_0, V_3_9_0):
        // * before V_2_17_0, `compression_level` param wasn't recognized (index creation fails);
        // * at or after V_3_9_0, both phases pick SQ so the switch is a no-op.
        if (isCompressionSupported(getBWCVersion()) == false || isPreFaissDefaultSwitch(getBWCVersion()) == false) {
            logger.info(
                "Skipping test — Faiss x8/x16 BQ→SQ default switch requires old cluster in [{}, {}), got: {}",
                Version.V_2_17_0,
                Version.V_3_9_0,
                getBWCVersion()
            );
            return;
        }
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        final String indexName = testIndex + indexSuffix;

        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
                // Bulk-ingest to force a single segment build with all NUM_DOCS vectors. `addKNNDocs`
                // refreshes per doc, which on 2.17 with BQ 4-bit / 2-bit fails the codec build
                // ("Number of vectors cannot be 0") on the initial single-vector segment.
                bulkAddKnnDocs(indexName, TEST_FIELD, buildIndexVectors(NUM_DOCS, DIMENSIONS), NUM_DOCS);
                flush(indexName, true);
                break;

            case MIXED:
                validateKNNSearch(indexName, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                assertPersistedCompressionAndMode(indexName, compressionLevel);
                break;

            case UPGRADED:
                validateKNNSearch(indexName, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                assertPersistedCompressionAndMode(indexName, compressionLevel);

                // Exercise post-upgrade writes and force-merge on the pre-gate BQ segments. Reuse
                // the same doc ids to keep the count at NUM_DOCS after refresh — bulkAddKnnDocs
                // always writes ids 0..N-1; the goal is to add fresh segments post-upgrade, not to
                // grow the doc set.
                bulkAddKnnDocs(indexName, TEST_FIELD, buildIndexVectors(NUM_DOCS, DIMENSIONS), NUM_DOCS);
                forceMergeKnnIndex(indexName);
                validateKNNSearch(indexName, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                deleteKNNIndex(indexName);
        }
    }

    private static float[][] buildIndexVectors(int count, int dim) {
        float[][] v = new float[count][dim];
        for (int i = 0; i < count; i++) {
            java.util.Arrays.fill(v[i], (float) i);
        }
        return v;
    }

    @SuppressWarnings("unchecked")
    private void assertPersistedCompressionAndMode(String indexName, CompressionLevel compressionLevel) throws Exception {
        Map<String, Object> mappings = getIndexMappingAsMap(indexName);
        Map<String, Object> properties = (Map<String, Object>) mappings.get(PROPERTIES);
        assertNotNull("Properties should not be null", properties);
        Map<String, Object> fieldProps = (Map<String, Object>) properties.get(TEST_FIELD);
        assertNotNull("Field properties should not be null", fieldProps);
        assertEquals(compressionLevel.getName(), fieldProps.get(COMPRESSION_LEVEL_PARAMETER));
        assertEquals(Mode.ON_DISK.getName(), fieldProps.get(MODE_PARAMETER));
    }

    /** True when the old cluster is strictly before the Faiss x8/x16 default switch (V_3_9_0). */
    private boolean isPreFaissDefaultSwitch(final Optional<String> bwcVersion) {
        if (bwcVersion.isEmpty()) {
            return false;
        }
        String versionString = bwcVersion.get().replace("-SNAPSHOT", "");
        return Version.fromString(versionString).before(Version.V_3_9_0);
    }

    private boolean isCompressionSupported(final Optional<String> bwcVersion) {
        if (bwcVersion.isEmpty()) {
            return false;
        }
        String versionString = bwcVersion.get().replace("-SNAPSHOT", "");
        return Version.fromString(versionString).onOrAfter(Version.V_2_17_0);
    }
}
