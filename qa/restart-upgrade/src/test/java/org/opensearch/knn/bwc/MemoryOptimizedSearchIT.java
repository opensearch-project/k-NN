/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.Version;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;

public class MemoryOptimizedSearchIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "target_field";
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 200;
    private static final int K = 5;

    public void testMemoryOptimizedSearchWithFaiss() throws Exception {
        doTestMemoryOptimizedSearch(FAISS_NAME);
    }

    public void testMemoryOptimizedSearchWithNmslib() throws Exception {
        doTestMemoryOptimizedSearch(NMSLIB_NAME);
    }

    private void doTestMemoryOptimizedSearch(final String engine) throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            // Create index and ingest some data
            createIndexAndIngestData(engine);

            // Force merge to a single segment.
            forceMergeKnnIndex(testIndex);
        } else {
            String versionString = getBWCVersion().get();
            versionString = versionString.replaceAll("-SNAPSHOT", "");
            final Version oldVersion = Version.fromString(versionString);

            // Turn on memory optimized search
            turnOnMemoryOptSearch();

            // Validate warm-up is done without an issue
            knnWarmup(Collections.singletonList(testIndex));

            if (oldVersion.compareTo(Version.V_2_17_0) < 0) {
                if (engine.equals(NMSLIB_NAME)) {
                    // Search should work regardless
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);

                    // Memory optimized search is applied, but NMSLIB should load off-heap index.
                    assertEquals(1, getTotalGraphsInCache());
                } else {
                    // For FAISS engine, indices created < 2.17 should throw an exception.
                    final float[] queryVector = new float[DIMENSION];
                    Arrays.fill(queryVector, (float) NUM_DOCS);

                    // Exception should be thrown
                    assertThrows(
                        ResponseException.class,
                        () -> searchKNNIndex(
                            testIndex,
                            KNNQueryBuilder.builder().k(K).methodParameters(null).fieldName(TEST_FIELD).vector(queryVector).build(),
                            K
                        )
                    );
                }
            } else {
                // Validate search, should be fine
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);

                if (engine.equals(NMSLIB_NAME)) {
                    // Memory optimized search is applied, but NMSLIB should load off-heap index.
                    assertEquals(1, getTotalGraphsInCache());
                } else {
                    // Memory optimized search is applied, no off-heap graph is expected
                    assertEquals(0, getTotalGraphsInCache());
                }
            }

            // Delete index
            deleteKNNIndex(testIndex);
        }
    }

    private void turnOnMemoryOptSearch() throws IOException {
        // Close index
        closeKNNIndex(testIndex);

        // Update settings
        OpenSearchRestTestCase.updateIndexSettings(testIndex, Settings.builder().put(KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true));

        // Reopen index again
        OpenSearchRestTestCase.openIndex(testIndex);
    }

    private void createIndexAndIngestData(final String engine) throws IOException {
        final Settings indexSettings = getKNNDefaultIndexSettings();

        // Create an index
        createKnnIndex(
            testIndex,
            indexSettings,
            createKnnIndexMapping(TEST_FIELD, DIMENSION, METHOD_HNSW, engine, SpaceType.L2.getValue())
        );

        // Ingest 200 docs
        addKNNDocs(testIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
    }
}
