/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.Version;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

/**
 * BWC integration test for ResolvedIndexSpec. Verifies that indices created before
 * ResolvedIndexSpec was introduced (with old-style encoder params) still work after
 * a restart upgrade to the new version.
 */
public class ResolvedIndexSpecIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 100;
    private static final int K = 10;

    /**
     * Tests that an HNSW index with SQ encoder bits=1 (1-bit quantization) created on the old
     * cluster is still queryable after a restart upgrade.
     */
    public void testHNSWSQOneBit_onOldClusterThenUpgraded_thenSucceed() throws Exception {
        if (isSQEncoderSupported(getBWCVersion()) == false) {
            logger.info("Skipping test as SQ encoder is not supported in version: {}", getBWCVersion());
            return;
        }
        String indexName = testIndex + "-sq-1bit";
        if (isRunningAgainstOldCluster()) {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .startObject("method")
                .field(NAME, "hnsw")
                .field(KNN_ENGINE, FAISS_NAME)
                .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
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

            createKnnIndex(indexName, builder.toString());
            indexTestData(indexName, TEST_FIELD, DIMENSION, NUM_DOCS);
        } else {
            queryAndValidate(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
            deleteKNNIndex(indexName);
        }
    }

    /**
     * Tests that an HNSW index with SQ encoder bits=16 (fp16 quantization) created on the old
     * cluster is still queryable after a restart upgrade.
     */
    public void testHNSWSQSixteenBit_onOldClusterThenUpgraded_thenSucceed() throws Exception {
        if (isSQEncoderSupported(getBWCVersion()) == false) {
            logger.info("Skipping test as SQ encoder is not supported in version: {}", getBWCVersion());
            return;
        }
        String indexName = testIndex + "-sq-16bit";
        if (isRunningAgainstOldCluster()) {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .startObject("method")
                .field(NAME, "hnsw")
                .field(KNN_ENGINE, FAISS_NAME)
                .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                .startObject(PARAMETERS)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, ENCODER_SQ)
                .startObject(PARAMETERS)
                .field(SQ_BITS, 16)
                .field(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject();

            createKnnIndex(indexName, builder.toString());
            indexTestData(indexName, TEST_FIELD, DIMENSION, NUM_DOCS);
        } else {
            queryAndValidate(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
            deleteKNNIndex(indexName);
        }
    }

    /**
     * Tests that an HNSW index with no encoder (flat/default) created on the old cluster
     * is still queryable after a restart upgrade.
     */
    public void testHNSWNoEncoder_onOldClusterThenUpgraded_thenSucceed() throws Exception {
        String indexName = testIndex + "-no-encoder";
        if (isRunningAgainstOldCluster()) {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(TEST_FIELD)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .startObject("method")
                .field(NAME, "hnsw")
                .field(KNN_ENGINE, FAISS_NAME)
                .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                .endObject()
                .endObject()
                .endObject()
                .endObject();

            createKnnIndex(indexName, builder.toString());
            indexTestData(indexName, TEST_FIELD, DIMENSION, NUM_DOCS);
        } else {
            queryAndValidate(indexName, TEST_FIELD, DIMENSION, NUM_DOCS, K);
            deleteKNNIndex(indexName);
        }
    }

    private boolean isSQEncoderSupported(final Optional<String> bwcVersion) {
        if (bwcVersion.isEmpty()) {
            return false;
        }
        String versionString = bwcVersion.get().replace("-SNAPSHOT", "");
        return Version.fromString(versionString).onOrAfter(Version.V_3_6_0);
    }

    private void indexTestData(String indexName, String fieldName, int dimension, int numDocs) throws Exception {
        for (int i = 0; i < numDocs; i++) {
            float[] vector = new float[dimension];
            Arrays.fill(vector, (float) i);
            addKnnDoc(indexName, Integer.toString(i), fieldName, vector);
        }
        refreshAllNonSystemIndices();
        assertEquals(numDocs, getDocCount(indexName));
    }

    private void queryAndValidate(String indexName, String fieldName, int dimension, int numDocs, int k) throws IOException,
        ParseException {
        float[] queryVector = new float[dimension];
        Arrays.fill(queryVector, (float) numDocs);

        Response searchResponse = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, queryVector, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(searchResponse.getEntity()), fieldName);
        assertEquals(k, results.size());
        for (int i = 0; i < k; i++) {
            assertEquals(numDocs - i - 1, Integer.parseInt(results.get(i).getDocId()));
        }
    }
}
