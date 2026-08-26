/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Optional;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
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
 * Restart-upgrade BWC coverage for Lucene HNSW SQ 2-bit (x16) and 4-bit (x8). Creates the
 * index and ingests docs on the old cluster (which must already be at
 * {@link org.opensearch.knn.common.KNNConstants#LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION}), then
 * verifies the index is searchable and accepts new docs after the restart.
 */
public class LuceneSQMultiBitCompressionIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    public void testRestartUpgrade_luceneSQ_x16() throws Exception {
        runRestartUpgradeForCompression(CompressionLevel.x16, "-x16");
    }

    public void testRestartUpgrade_luceneSQ_x8() throws Exception {
        runRestartUpgradeForCompression(CompressionLevel.x8, "-x8");
    }

    private void runRestartUpgradeForCompression(CompressionLevel compressionLevel, String indexSuffix) throws Exception {
        if (isLuceneSQMultiBitSupported(getBWCVersion()) == false) {
            logger.info(
                "Skipping test — Lucene HNSW SQ {} requires old cluster >= {}, got: {}",
                compressionLevel.getName(),
                LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION,
                getBWCVersion()
            );
            return;
        }

        final String indexName = testIndex + indexSuffix;

        if (isRunningAgainstOldCluster()) {
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
        } else {
            validateKNNSearch(indexName, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            addKNNDocs(indexName, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(indexName);
            validateKNNSearch(indexName, TEST_FIELD, DIMENSIONS, 2 * NUM_DOCS, K);
            deleteKNNIndex(indexName);
        }
    }

    private boolean isLuceneSQMultiBitSupported(final Optional<String> bwcVersion) {
        if (bwcVersion.isEmpty()) {
            return false;
        }
        String versionString = bwcVersion.get().replace("-SNAPSHOT", "");
        return Version.fromString(versionString).onOrAfter(LUCENE_HNSW_SQ_2BIT_4BIT_MIN_VERSION);
    }
}
