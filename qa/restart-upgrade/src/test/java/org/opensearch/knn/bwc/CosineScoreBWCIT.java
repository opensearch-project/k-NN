/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * BWC test verifying cosine similarity scores remain identical across versions.
 *
 * <p>Phase 1 (old cluster / 3.6): Creates a cosine FAISS index with fixed vectors,
 * searches and verifies scores match precomputed expected values.
 *
 * <p>Phase 2 (upgraded cluster / main): Enables memory optimized search on the same
 * index, searches and verifies the same expected scores.
 */
public class CosineScoreBWCIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "vec";
    private static final int DIMENSION = 3;
    private static final int K = 3;

    // Fixed unit vectors producing distinct cosine similarities with QUERY
    private static final float[][] VECTORS = {
        { 1.0f, 0.0f, 0.0f },   // doc 0
        { 0.0f, 1.0f, 0.0f },   // doc 1
        { 0.6f, 0.8f, 0.0f },   // doc 2
    };
    private static final float[] QUERY = { 1.0f, 0.0f, 0.0f };

    // Precomputed expected scores: KNNVectorSimilarityFunction.COSINE.compare(query, vec) = (1 + cosine) / 2
    // doc0: (1+1.0)/2=1.0, doc2: (1+0.6)/2=0.8, doc1: (1+0.0)/2=0.5
    // Sorted descending by score
    private static final String[] EXPECTED_DOC_IDS = { "0", "2", "1" };
    private static final float[] EXPECTED_SCORES = {
        KNNVectorSimilarityFunction.COSINE.compare(QUERY, VECTORS[0]),
        KNNVectorSimilarityFunction.COSINE.compare(QUERY, VECTORS[2]),
        KNNVectorSimilarityFunction.COSINE.compare(QUERY, VECTORS[1]), };

    public void testCosineScoreConsistencyAcrossVersions() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            // Create cosine index and ingest fixed vectors
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSION, METHOD_HNSW, FAISS_NAME, SpaceType.COSINESIMIL.getValue())
            );
            for (int i = 0; i < VECTORS.length; i++) {
                addKnnDoc(testIndex, Integer.toString(i), TEST_FIELD, toFloatObject(VECTORS[i]));
            }
            forceMergeKnnIndex(testIndex);

            // Verify scores on old cluster
            verifyScores("old cluster");
        } else {
            // Enable memory optimized search on the upgraded cluster
            closeKNNIndex(testIndex);
            OpenSearchRestTestCase.updateIndexSettings(
                testIndex,
                Settings.builder().put(KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true)
            );
            OpenSearchRestTestCase.openIndex(testIndex);
            knnWarmup(Collections.singletonList(testIndex));

            // Verify same scores after upgrade with memory optimized search
            verifyScores("upgraded cluster (memory optimized)");
        }
    }

    private void verifyScores(String phase) throws Exception {
        Response response = searchKNNIndex(testIndex, KNNQueryBuilder.builder().k(K).fieldName(TEST_FIELD).vector(QUERY).build(), K);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), TEST_FIELD);
        assertEquals(K, results.size());

        Map<String, Float> scoresByDocId = results.stream().collect(Collectors.toMap(KNNResult::getDocId, KNNResult::getScore));

        for (int i = 0; i < K; i++) {
            Float actual = scoresByDocId.get(EXPECTED_DOC_IDS[i]);
            assertNotNull("Doc " + EXPECTED_DOC_IDS[i] + " missing on " + phase, actual);
            assertEquals("Score mismatch for doc " + EXPECTED_DOC_IDS[i] + " on " + phase, EXPECTED_SCORES[i], actual, 1e-4f);
        }
    }

    private static Float[] toFloatObject(float[] v) {
        Float[] result = new Float[v.length];
        for (int i = 0; i < v.length; i++)
            result[i] = v[i];
        return result;
    }
}
