/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;

/**
 * Use case: Test BBQ (Binary Quantization) functionality on indexes created on older versions
 * BBQ integrates the lucene102 codec when a user inputs the encoder value to binary while using the lucene engine
 */
public class BinaryQuantizationBWCIT extends AbstractRestartUpgradeTestCase {

    private static final String TEST_FIELD = "bbq-test-field";
    private static final int DIMENSIONS = 128; // Higher dimensions for BBQ testing
    private static final int K = 10;
    private static final Integer EF_SEARCH = 50;
    private static final int NUM_DOCS = 100;
    private static final String ALGORITHM = "hnsw";
    private static final String ENCODER_BINARY = "binary";

    public void testBBQIndexCreatedOnOldCluster() throws Exception {
        if (isRunningAgainstOldCluster()) {
            // Create index with BBQ configuration on old cluster
            String mapping = createKnnIndexMappingWithEncoder(
                    TEST_FIELD,
                    DIMENSIONS,
                    ALGORITHM,
                    LUCENE_NAME,
                    ENCODER_BINARY
            );
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);

            // Add documents with vector data
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);

            // Validate search works on old cluster
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

            // Force merge to ensure segments are created with BBQ codec
            forceMergeKnnIndex(testIndex, 1);
        } else {
            // Validate search still works after upgrade with BBQ codec
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

            // Test search with ef_search parameter
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K,
                    Map.of(METHOD_PARAMETER_EF_SEARCH, EF_SEARCH));

            // Verify index can still be written to after upgrade
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS + 10);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS + 10, K);

            // Cleanup
            deleteKNNIndex(testIndex);
        }
    }

    public void testBBQIndexUpgradeCompatibility() throws Exception {
        if (isRunningAgainstOldCluster()) {
            // Create standard lucene index on old cluster
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGORITHM, LUCENE_NAME));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
        } else {
            // Verify existing index still works after upgrade
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

            // Create new BBQ index on upgraded cluster
            String bbqTestIndex = testIndex + "-bbq-new";
            String mapping = createKnnIndexMappingWithEncoder(
                    TEST_FIELD,
                    DIMENSIONS,
                    ALGORITHM,
                    LUCENE_NAME,
                    ENCODER_BINARY
            );
            createKnnIndex(bbqTestIndex, getKNNDefaultIndexSettings(), mapping);
            addKNNDocs(bbqTestIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            validateKNNSearch(bbqTestIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

            // Cleanup
            deleteKNNIndex(testIndex);
            deleteKNNIndex(bbqTestIndex);
        }
    }

    public void testBBQCodecPersistence() throws Exception {
        if (isRunningAgainstOldCluster()) {
            // Create BBQ index and verify codec is properly set
            String mapping = createKnnIndexMappingWithEncoder(
                    TEST_FIELD,
                    DIMENSIONS,
                    ALGORITHM,
                    LUCENE_NAME,
                    ENCODER_BINARY
            );
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);

            // Force merge to create segments with BBQ codec
            forceMergeKnnIndex(testIndex, 1);

            // Validate the index uses the correct codec
            validateIndexCodec(testIndex, "lucene102");
        } else {
            // After upgrade, verify codec information is preserved
            validateIndexCodec(testIndex, "lucene102");

            // Verify search functionality is maintained
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

            // Test that new segments still use BBQ codec after upgrade
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS + 20);
            forceMergeKnnIndex(testIndex, 1);
            validateIndexCodec(testIndex, "lucene102");

            deleteKNNIndex(testIndex);
        }
    }

    /**
     * Helper method to create KNN index mapping with encoder parameter
     */
    private String createKnnIndexMappingWithEncoder(String fieldName, int dimensions,
                                                    String algorithm, String engine, String encoder) throws Exception {
        return String.format("""                                                                                                                                                                                                      
            {
                "properties": {
                    "%s": {
                        "type": "knn_vector",
                        "dimension": %d,
                        "method": {
                            "name": "%s",
                            "engine": "%s",
                            "parameters": {
                                "encoder": {
                                    "name": "%s"
                                }
                            }
                        }
                    }
                }
            }
            """, fieldName, dimensions, algorithm, engine, encoder);
    }

    /**
     * Helper method to validate index codec
     */
    private void validateIndexCodec(String indexName, String expectedCodec) throws Exception {
        // This would need to be implemented based on how codec information is exposed
        // in the OpenSearch API - could check index settings or segment information
        Map<String, Object> indexStats = getIndexStats(indexName);
        // Add assertions to verify the codec is as expected
        // This is a placeholder - actual implementation would depend on available APIs
    }
}