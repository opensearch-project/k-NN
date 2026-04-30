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

import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

/**
 * Rolling-upgrade BwC test for Faiss SQ FP16 encoder.
 * Creates a legacy SQ FP16 index (without bits) on the old cluster, verifies it survives
 * rolling upgrade, then creates a new index with bits=16 on the upgraded cluster.
 */
public class FaissSQIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSION = 128;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    public void testHNSWSQFP16_rollingUpgrade_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                // Create legacy SQ FP16 index without bits param (pre-3.6.0 format)
                createLegacySQFP16Index(testIndex);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);
                break;
            case MIXED:
                int totalDocs = isFirstMixedRound() ? 2 * NUM_DOCS : 3 * NUM_DOCS;
                int docId = isFirstMixedRound() ? NUM_DOCS : 2 * NUM_DOCS;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSION, docId, NUM_DOCS);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSION, totalDocs, K);
                break;
            case UPGRADED:
                addKNNDocs(testIndex, TEST_FIELD, DIMENSION, 3 * NUM_DOCS, NUM_DOCS);
                forceMergeKnnIndex(testIndex);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSION, 4 * NUM_DOCS, K);
                deleteKNNIndex(testIndex);
        }
    }

    public void testHNSWSQFP16WithBits_onUpgradedCluster_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        String upgradedIndex = testIndex + "-upgraded";
        switch (getClusterType()) {
            case OLD:
            case MIXED:
                break;
            case UPGRADED:
                // Create new index with bits=16 on fully upgraded cluster
                createSQFP16WithBitsIndex(upgradedIndex);
                addKNNDocs(upgradedIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                validateKNNSearch(upgradedIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);
                deleteKNNIndex(upgradedIndex);
        }
    }

    public void testLegacySQFP16WithClip_rollingUpgrade_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        String clipIndex = testIndex + "-clip";
        switch (getClusterType()) {
            case OLD:
                createLegacySQFP16WithClipIndex(clipIndex);
                addKNNDocs(clipIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                validateKNNSearch(clipIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);
                break;
            case MIXED:
                int clipTotalDocs = isFirstMixedRound() ? 2 * NUM_DOCS : 3 * NUM_DOCS;
                addKNNDocs(clipIndex, TEST_FIELD, DIMENSION, isFirstMixedRound() ? NUM_DOCS : 2 * NUM_DOCS, NUM_DOCS);
                validateKNNSearch(clipIndex, TEST_FIELD, DIMENSION, clipTotalDocs, K);
                break;
            case UPGRADED:
                addKNNDocs(clipIndex, TEST_FIELD, DIMENSION, 3 * NUM_DOCS, NUM_DOCS);
                forceMergeKnnIndex(clipIndex);
                validateKNNSearch(clipIndex, TEST_FIELD, DIMENSION, 4 * NUM_DOCS, K);
                deleteKNNIndex(clipIndex);
        }
    }

    public void testLegacySQFP16WithOnDisk_rollingUpgrade_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        String onDiskIndex = testIndex + "-ondisk";
        switch (getClusterType()) {
            case OLD:
                createLegacySQFP16WithOnDiskIndex(onDiskIndex);
                addKNNDocs(onDiskIndex, TEST_FIELD, DIMENSION, 0, NUM_DOCS);
                validateKNNSearch(onDiskIndex, TEST_FIELD, DIMENSION, NUM_DOCS, K);
                break;
            case MIXED:
                int onDiskTotalDocs = isFirstMixedRound() ? 2 * NUM_DOCS : 3 * NUM_DOCS;
                addKNNDocs(onDiskIndex, TEST_FIELD, DIMENSION, isFirstMixedRound() ? NUM_DOCS : 2 * NUM_DOCS, NUM_DOCS);
                validateKNNSearch(onDiskIndex, TEST_FIELD, DIMENSION, onDiskTotalDocs, K);
                break;
            case UPGRADED:
                addKNNDocs(onDiskIndex, TEST_FIELD, DIMENSION, 3 * NUM_DOCS, NUM_DOCS);
                forceMergeKnnIndex(onDiskIndex);
                validateKNNSearch(onDiskIndex, TEST_FIELD, DIMENSION, 4 * NUM_DOCS, K);
                deleteKNNIndex(onDiskIndex);
        }
    }

    private void createLegacySQFP16Index(String indexName) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field(NAME, "hnsw")
            .field(KNN_ENGINE, FAISS_NAME)
            .field("space_type", SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, builder.toString());
    }

    private void createSQFP16WithBitsIndex(String indexName) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field(NAME, "hnsw")
            .field(KNN_ENGINE, FAISS_NAME)
            .field("space_type", SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .field(SQ_BITS, 16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, builder.toString());
    }

    private void createLegacySQFP16WithClipIndex(String indexName) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject("method")
            .field(NAME, "hnsw")
            .field(KNN_ENGINE, FAISS_NAME)
            .field("space_type", SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .field(FAISS_SQ_CLIP, true)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());
    }

    private void createLegacySQFP16WithOnDiskIndex(String indexName) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(TEST_FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "on_disk")
            .startObject("method")
            .field(NAME, "hnsw")
            .field(KNN_ENGINE, FAISS_NAME)
            .field("space_type", SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());
    }

    /**
     * Validates new mapping scenarios on the fully upgraded cluster.
     */
    public void testSQMappingValidation_onUpgradedCluster_thenSucceed() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
            case MIXED:
                break;
            case UPGRADED:
                // bits=1 without type should succeed
                String bits1Index = testIndex + "-bits1";
                XContentBuilder bits1Builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(TEST_FIELD)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .startObject("method")
                    .field(NAME, "hnsw")
                    .field(KNN_ENGINE, FAISS_NAME)
                    .field("space_type", SpaceType.L2.getValue())
                    .startObject(PARAMETERS)
                    .startObject(METHOD_ENCODER_PARAMETER)
                    .field(NAME, ENCODER_SQ)
                    .startObject(PARAMETERS)
                    .field(SQ_BITS, 1)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject();
                createKnnIndex(bits1Index, bits1Builder.toString());
                deleteKNNIndex(bits1Index);

                // sq without bits on 3.6.0+ should be rejected
                String noBitsMapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(TEST_FIELD)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .startObject("method")
                    .field(NAME, "hnsw")
                    .field(KNN_ENGINE, FAISS_NAME)
                    .startObject(PARAMETERS)
                    .startObject(METHOD_ENCODER_PARAMETER)
                    .field(NAME, ENCODER_SQ)
                    .startObject(PARAMETERS)
                    .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                expectThrows(ResponseException.class, () -> createKnnIndex(testIndex + "-nobits", noBitsMapping));

                // bits=1 with type=fp16 should be rejected
                String bits1TypeMapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject("properties")
                    .startObject(TEST_FIELD)
                    .field("type", "knn_vector")
                    .field("dimension", DIMENSION)
                    .startObject("method")
                    .field(NAME, "hnsw")
                    .field(KNN_ENGINE, FAISS_NAME)
                    .startObject(PARAMETERS)
                    .startObject(METHOD_ENCODER_PARAMETER)
                    .field(NAME, ENCODER_SQ)
                    .startObject(PARAMETERS)
                    .field(SQ_BITS, 1)
                    .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                expectThrows(ResponseException.class, () -> createKnnIndex(testIndex + "-bits1type", bits1TypeMapping));
        }
    }
}
