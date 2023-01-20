/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.SpaceType;

import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_M_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class IndexingIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int M = 50;
    private static final int EF_CONSTRUCTION = 1024;
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
            validateKNNIndexingOnUpgrade();
        }
    }

    // Custom Legacy Field Mapping
    // space_type : "linf", engine : "nmslib", m : 2, ef_construction : 2
    public void testKNNIndexCustomLegacyFieldMapping() throws Exception {

        // When the cluster is in old version, create a KNN index with custom legacy field mapping settings
        // and add documents into that index
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                createKNNIndexCustomLegacyFieldMappingSettings(
                    SpaceType.LINF,
                    KNN_ALGO_PARAM_M_MIN_VALUE,
                    KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE
                ),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNIndexingOnUpgrade();
        }
    }

    // Default Method Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKNNIndexDefaultMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKNNIndexMethodFieldMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNIndexingOnUpgrade();
        }
    }

    // Custom Method Field Mapping
    // space_type : "inner_product", engine : "faiss", m : 50, ef_construction : 1024
    public void testKNNIndexCustomMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKNNIndexCustomMethodFieldMapping(TEST_FIELD, DIMENSIONS, SpaceType.INNER_PRODUCT, FAISS_NAME, M, EF_CONSTRUCTION)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        } else {
            validateKNNIndexingOnUpgrade();
        }
    }

    // test null parameters
    public void testNullParametersOnUpgrade() throws Exception {
        if (isRunningAgainstOldCluster()) {
            String mapping = Strings.toString(
                XContentFactory.jsonBuilder()
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
            );

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
        } else {
            deleteKNNIndex(testIndex);
        }
    }

    // test empty parameters
    public void testEmptyParametersOnUpgrade() throws Exception {
        if (isRunningAgainstOldCluster()) {
            String mapping = Strings.toString(
                XContentFactory.jsonBuilder()
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
            );

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
        } else {
            deleteKNNIndex(testIndex);
        }
    }

    // test no parameters
    public void testNoParametersOnUpgrade() throws Exception {
        if (isRunningAgainstOldCluster()) {
            String mapping = Strings.toString(
                XContentFactory.jsonBuilder()
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
            );

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
        } else {
            deleteKNNIndex(testIndex);
        }
    }

    // KNN indexing tests when the cluster is upgraded to latest version
    public void validateKNNIndexingOnUpgrade() throws Exception {
        QUERY_COUNT = NUM_DOCS;
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
        clearCache();
        DOC_ID = NUM_DOCS;
        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
        QUERY_COUNT = QUERY_COUNT + NUM_DOCS;
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
        forceMergeKnnIndex(testIndex);
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
        deleteKNNIndex(testIndex);
    }
}
