/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.junit.Assert;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.List;
import java.util.Map;

import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_M_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
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
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class IndexingIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
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
}
