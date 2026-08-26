/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;

/**
 * Rolling-upgrade BWC coverage for Lucene HNSW SQ 2-bit (x16) and 4-bit (x8). Creates the
 * index on the old cluster (which must already be at
 * {@link org.opensearch.knn.common.KNNConstants#LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION}), then
 * verifies the index is searchable in the mixed and upgraded phases.
 *
 * <p>Older BWC versions do not accept x16/x8 on Lucene HNSW, so the test skips when the
 * old cluster predates the gate.
 */
public class LuceneSQMultiBitCompressionIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    public void testRollingUpgrade_luceneSQ_x16() throws Exception {
        runRollingUpgradeForCompression(CompressionLevel.x16, "-x16");
    }

    public void testRollingUpgrade_luceneSQ_x8() throws Exception {
        runRollingUpgradeForCompression(CompressionLevel.x8, "-x8");
    }

    @SuppressWarnings("unchecked")
    private void runRollingUpgradeForCompression(CompressionLevel compressionLevel, String indexSuffix) throws Exception {
        if (isLuceneSQMultiBitSupported(getBWCVersion()) == false) {
            logger.info(
                "Skipping test — Lucene HNSW SQ {}/{} requires old cluster >= {}, got: {}",
                compressionLevel.getName(),
                LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION,
                LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION,
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
                    .startObject("method")
                    .field(NAME, METHOD_HNSW)
                    .field(KNN_ENGINE, LUCENE_NAME)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(indexName, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(indexName, true);
                break;

            case MIXED:
                validateKNNSearch(indexName, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                Map<String, Object> mixedMappings = getIndexMappingAsMap(indexName);
                Map<String, Object> mixedProperties = (Map<String, Object>) mixedMappings.get(PROPERTIES);
                assertNotNull("Properties should not be null", mixedProperties);
                Map<String, Object> mixedFieldProps = (Map<String, Object>) mixedProperties.get(TEST_FIELD);
                assertNotNull("Field properties should not be null", mixedFieldProps);
                assertEquals(compressionLevel.getName(), mixedFieldProps.get(COMPRESSION_LEVEL_PARAMETER));
                break;

            case UPGRADED:
                validateKNNSearch(indexName, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

                Map<String, Object> upgradedMappings = getIndexMappingAsMap(indexName);
                Map<String, Object> upgradedProperties = (Map<String, Object>) upgradedMappings.get(PROPERTIES);
                assertNotNull("Properties should not be null after upgrade", upgradedProperties);
                Map<String, Object> upgradedFieldProps = (Map<String, Object>) upgradedProperties.get(TEST_FIELD);
                assertNotNull("Field properties should not be null after upgrade", upgradedFieldProps);
                assertEquals(compressionLevel.getName(), upgradedFieldProps.get(COMPRESSION_LEVEL_PARAMETER));

                deleteKNNIndex(indexName);
        }
    }

    /**
     * True when the BWC old-cluster version is at or after
     * {@link org.opensearch.knn.common.KNNConstants#LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION} —
     * i.e., the old cluster already accepts x16/x8 on Lucene HNSW.
     */
    private boolean isLuceneSQMultiBitSupported(final Optional<String> bwcVersion) {
        if (bwcVersion.isEmpty()) {
            return false;
        }
        String versionString = bwcVersion.get().replace("-SNAPSHOT", "");
        return Version.fromString(versionString).onOrAfter(LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION);
    }
}
