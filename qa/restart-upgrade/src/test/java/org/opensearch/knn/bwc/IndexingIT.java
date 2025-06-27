/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.Assert;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.containsString;
import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_M_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class IndexingIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final String TEST_FIELD2 = "test-field2";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int M = 50;
    private static final int EF_CONSTRUCTION = 1024;
    private static final int EF_SEARCH = 200;
    private static final int NUM_DOCS = 10;
    private static int QUERY_COUNT = 0;

    // Default Legacy Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKNNIndexDefaultLegacyFieldMapping() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            // update index setting to allow build graph always since we test graph count that are loaded into memory
            updateIndexSettings(testIndex, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0));
            validateKNNIndexingOnUpgrade(NUM_DOCS);
        }
    }

    // ensure that index is created using legacy mapping in old cluster, and, then, add docs to both old and new cluster.
    // when search is requested on new cluster it should return all docs irrespective of cluster.
    public void testKNNIndexDefaultEngine() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, 5);
            // Flush to ensure that index is not re-indexed when node comes back up
            flush(testIndex, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 5, 5);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 5, 5);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 10, 10);
            deleteKNNIndex(testIndex);
        }
    }

    // Ensure that when segments created with old mapping are forcemerged in new cluster, they
    // succeed
    public void testKNNIndexDefaultLegacyFieldMappingForceMerge() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, 100);
            // Flush to ensure that index is not re-indexed when node comes back up
            flush(testIndex, true);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 100, K);
        } else {
            validateKNNIndexingOnUpgrade(100);
        }
    }

    public void testKNNIndexFaissForceMerge() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, FAISS_NAME));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, 100);
            // Flush to ensure that index is not re-indexed when node comes back up
            flush(testIndex, true);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 100, K);
        } else {
            validateKNNIndexingOnUpgrade(100);
        }
    }

    public void testKNNIndexLuceneForceMerge() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, LUCENE_NAME)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, 100);
            // Flush to ensure that index is not re-indexed when node comes back up
            flush(testIndex, true);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 100, K);
        } else {
            validateKNNIndexingOnUpgrade(100);
        }
    }

    public void testKNNIndexSettingImmutableAfterUpgrade() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        } else {
            Exception ex = expectThrows(
                ResponseException.class,
                () -> updateIndexSettings(testIndex, Settings.builder().put(KNNSettings.KNN_INDEX, false))
            );
            assertThat(ex.getMessage(), containsString("Can't update non dynamic settings [[index.knn]] for open indices"));

            closeIndex(testIndex);

            ex = expectThrows(
                ResponseException.class,
                () -> updateIndexSettings(testIndex, Settings.builder().put(KNNSettings.KNN_INDEX, false))
            );
            assertThat(ex.getMessage(), containsString(String.format("final %s setting [index.knn], not updateable", testIndex)));
        }
    }

    public void testKNNIndexLuceneByteVector() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, LUCENE_NAME, SpaceType.L2.getValue(), true, VectorDataType.BYTE)
            );
            addKNNByteDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, 50);
            // Flush to ensure that index is not re-indexed when node comes back up
            flush(testIndex, true);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 50, 5);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 50, 5);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 50, 25);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 75, 5);
            deleteKNNIndex(testIndex);
        }
    }

    public void testKNNIndexLuceneQuantization() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        int k = 4;
        int dimension = 2;

        if (isRunningAgainstOldCluster()) {
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, dimension)
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
                .field(KNN_ENGINE, LUCENE_NAME)
                .startObject(PARAMETERS)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, ENCODER_SQ)
                .endObject()
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, 256)
                .field(METHOD_PARAMETER_M, 16)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);

            Float[] vector1 = { -10.6f, 25.48f };
            Float[] vector2 = { -10.8f, 25.48f };
            Float[] vector3 = { -11.0f, 25.48f };
            Float[] vector4 = { -11.2f, 25.48f };
            addKnnDoc(testIndex, "1", TEST_FIELD, vector1);
            addKnnDoc(testIndex, "2", TEST_FIELD, vector2);
            addKnnDoc(testIndex, "3", TEST_FIELD, vector3);
            addKnnDoc(testIndex, "4", TEST_FIELD, vector4);

            float[] queryVector = { -10.5f, 25.48f };
            Response searchResponse = searchKNNIndex(testIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, k), k);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), TEST_FIELD);
            assertEquals(k, results.size());
            for (int i = 0; i < k; i++) {
                assertEquals(k - i, Integer.parseInt(results.get(i).getDocId()));
            }
        } else {
            float[] queryVector = { -10.5f, 25.48f };
            Response searchResponse = searchKNNIndex(testIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, k), k);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), TEST_FIELD);
            assertEquals(k, results.size());
            for (int i = 0; i < k; i++) {
                assertEquals(k - i, Integer.parseInt(results.get(i).getDocId()));
            }
            deleteKNNIndex(testIndex);
        }
    }

    // Ensure bwc works for binary force merge
    public void testKNNIndexBinaryForceMerge() throws Exception {
        int dimension = 40;

        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexMapping(
                    TEST_FIELD,
                    dimension,
                    METHOD_HNSW,
                    KNNEngine.FAISS.getName(),
                    SpaceType.HAMMING.getValue(),
                    true,
                    VectorDataType.BINARY
                )
            );
            addKNNByteDocs(testIndex, TEST_FIELD, dimension / 8, DOC_ID, 100);
            // Flush to ensure that index is not re-indexed when node comes back up
            flush(testIndex, true);
        } else {
            forceMergeKnnIndex(testIndex);
        }
    }

    // Custom Legacy Field Mapping
    // space_type : "innerproduct", engine : "nmslib", m : 2, ef_construction : 2
    public void testKNNIndexCustomLegacyFieldMapping() throws Exception {

        // When the cluster is in old version, create a KNN index with custom legacy field mapping settings
        // and add documents into that index
        if (isRunningAgainstOldCluster()) {
            Settings.Builder indexMappingSettings = createKNNIndexCustomLegacyFieldMappingIndexSettingsBuilder(
                SpaceType.INNER_PRODUCT,
                KNN_ALGO_PARAM_M_MIN_VALUE,
                KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE
            );
            if (isApproximateThresholdSupported(getBWCVersion())) {
                indexMappingSettings.put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0);
            }
            createKnnIndex(testIndex, indexMappingSettings.build(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNIndexingOnUpgrade(NUM_DOCS);
        }
    }

    // Default Method Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKNNIndexDefaultMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKNNIndexMethodFieldMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNIndexingOnUpgrade(NUM_DOCS);
        }
    }

    // Custom Method Field Mapping
    // space_type : "inner_product", engine : "faiss", m : 50, ef_construction : 1024, ef_search : 200
    public void testKNNIndexCustomMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKNNIndexCustomMethodFieldMapping(
                    TEST_FIELD,
                    DIMENSIONS,
                    SpaceType.INNER_PRODUCT,
                    FAISS_NAME,
                    M,
                    EF_CONSTRUCTION,
                    EF_SEARCH
                )
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateCustomMethodFieldMappingAfterUpgrade();
            validateKNNIndexingOnUpgrade(NUM_DOCS);
        }
    }

    private void validateCustomMethodFieldMappingAfterUpgrade() throws Exception {
        final Map<String, Object> indexMappings = getIndexMappingAsMap(testIndex);
        final Map<String, Object> properties = (Map<String, Object>) indexMappings.get(PROPERTIES);
        final Map<String, Object> knnMethod = ((Map<String, Object>) ((Map<String, Object>) properties.get(TEST_FIELD)).get(KNN_METHOD));
        final Map<String, Object> methodParameters = (Map<String, Object>) knnMethod.get(PARAMETERS);

        Assert.assertEquals(METHOD_HNSW, knnMethod.get(NAME));
        Assert.assertEquals(SpaceType.INNER_PRODUCT.getValue(), knnMethod.get(METHOD_PARAMETER_SPACE_TYPE));
        Assert.assertEquals(FAISS_NAME, knnMethod.get(KNN_ENGINE));
        Assert.assertEquals(EF_CONSTRUCTION, ((Integer) methodParameters.get(METHOD_PARAMETER_EF_CONSTRUCTION)).intValue());
        Assert.assertEquals(EF_SEARCH, ((Integer) methodParameters.get(METHOD_PARAMETER_EF_SEARCH)).intValue());
        Assert.assertEquals(M, ((Integer) methodParameters.get(METHOD_PARAMETER_M)).intValue());
    }

    // test null parameters
    public void testNullParametersOnUpgrade() throws Exception {
        if (isRunningAgainstOldCluster()) {
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(TEST_FIELD)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, String.valueOf(DIMENSIONS))
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .field(PARAMETERS, (String) null)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
        } else {
            deleteKNNIndex(testIndex);
        }
    }

    // test empty parameters
    public void testEmptyParametersOnUpgrade() throws Exception {
        if (isRunningAgainstOldCluster()) {
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(TEST_FIELD)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, String.valueOf(DIMENSIONS))
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .field(PARAMETERS, "")
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
        } else {
            deleteKNNIndex(testIndex);
        }
    }

    // test no parameters
    public void testNoParametersOnUpgrade() throws Exception {
        if (isRunningAgainstOldCluster()) {
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(TEST_FIELD)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, String.valueOf(DIMENSIONS))
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
        } else {
            deleteKNNIndex(testIndex);
        }
    }

    // KNN indexing tests when the cluster is upgraded to latest version
    public void validateKNNIndexingOnUpgrade(int numOfDocs) throws Exception {
        updateIndexSettings(testIndex, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0));
        forceMergeKnnIndex(testIndex);
        QUERY_COUNT = numOfDocs;
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
        clearCache(List.of(testIndex));
        DOC_ID = numOfDocs;
        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
        forceMergeKnnIndex(testIndex);
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
        deleteKNNIndex(testIndex);
    }

    /**
     * Test to verify that NMSLIB index creation is blocked in OpenSearch 3.0.0 and later,
     * while ensuring backward compatibility (BWC) for existing indexes created in OpenSearch 2.19.
     *
     * @throws Exception if any unexpected error occurs during the test execution.
     */
    public void testBlockNMSLIBIndexCreationPost3_0_0() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, KNNEngine.NMSLIB.getName())
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            // Flush to ensure the index persists after upgrade
            flush(testIndex, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            expectThrows(
                ResponseException.class,
                () -> createKnnIndex(
                    testIndex + "_new",
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, KNNEngine.NMSLIB.getName())
                )
            );
            deleteKNNIndex(testIndex);
        }
    }

    public void testBlockModeAndCompressionBefore2_17_0() throws IOException {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, KNNEngine.NMSLIB.getName())
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
            flush(testIndex, true);
        } else {
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(TEST_FIELD2)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, String.valueOf(DIMENSIONS))
                .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
                .endObject()
                .endObject()
                .endObject()
                .toString();
            expectThrows(ResponseException.class, () -> putMappingRequest(testIndex, mapping));
            deleteKNNIndex(testIndex);
        }
    }

    public void testRandomRotationBWC() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        int k = 4;
        int dimension = 8;

        if (isRunningAgainstOldCluster()) {
            // In old cluster (2.20), create index with binary quantization but without random rotation
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, dimension)
                .startObject(KNN_METHOD)
                .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                .field(KNN_ENGINE, FAISS_NAME)
                .field(NAME, METHOD_HNSW)
                .startObject(PARAMETERS)
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, 256)
                .field(METHOD_PARAMETER_M, 16)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "binary")
                .startObject(PARAMETERS)
                .field("bits", 1)
                // No random rotation parameter in old version
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);

            // Add test vectors
            Float[] vector1 = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            Float[] vector2 = { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
            addKnnDoc(testIndex, "1", TEST_FIELD, vector1);
            addKnnDoc(testIndex, "2", TEST_FIELD, vector2);
            flush(testIndex, true);

        } else {
            // In new cluster (3.1), test searching existing index
            float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            Response searchResponse = searchKNNIndex(testIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, k), k);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), TEST_FIELD);
            assertEquals(2, results.size());

            // Create new index with random rotation enabled
            String newIndex = testIndex + "_random_rotation";
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, dimension)
                .startObject(KNN_METHOD)
                .field(NAME, METHOD_HNSW)
                .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                .field(KNN_ENGINE, FAISS_NAME)
                .startObject(PARAMETERS)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "binary")
                .startObject(PARAMETERS)
                .field("bits", 1)
                .field("random_rotation", true)
                .endObject()
                .endObject()
                .field(METHOD_PARAMETER_EF_CONSTRUCTION, 256)
                .field(METHOD_PARAMETER_M, 16)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();
            createKnnIndex(newIndex, getKNNDefaultIndexSettings(), mapping);

            // Add vectors to new index
            Float[] vector1 = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            Float[] vector2 = { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
            addKnnDoc(newIndex, "1", TEST_FIELD, vector1);
            addKnnDoc(newIndex, "2", TEST_FIELD, vector2);

            // Test search works with random rotation
            searchResponse = searchKNNIndex(newIndex, new KNNQueryBuilder(TEST_FIELD, queryVector, k), k);
            results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), TEST_FIELD);
            assertEquals(2, results.size());

            // Clean up
            deleteKNNIndex(testIndex);
            deleteKNNIndex(newIndex);
        }
    }
}
