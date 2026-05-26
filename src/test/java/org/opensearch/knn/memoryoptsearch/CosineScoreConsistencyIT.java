/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE;

/**
 * Verifies that cosine similarity scores are identical across three search paths:
 * 1. Standard FAISS search (HNSW graph, no memory optimization)
 * 2. Memory optimized search (reads FAISS index via Lucene Directory)
 * 3. Exact search (brute-force scoring, no graph traversal)
 *
 * This guards against regressions from the refactor that switched from
 * MAXIMUM_INNER_PRODUCT + post-conversion to direct COSINE scoring.
 */
public class CosineScoreConsistencyIT extends KNNRestTestCase {
    private static final String STANDARD_INDEX = "cosine_standard";
    private static final String MEM_OPT_INDEX = "cosine_mem_opt";
    private static final String EXACT_INDEX = "cosine_exact";
    private static final String FIELD = "vec";
    private static final int DIMENSION = 16;
    private static final int NUM_DOCS = 100;
    private static final int K = 10;

    @SneakyThrows
    public void testCosineScoresMatchBetweenStandardAndMemoryOptimizedSearch() {
        // Generate random normalized vectors
        final float[][] vectors = generateNormalizedVectors(NUM_DOCS, DIMENSION);
        final float[] query = normalizeVector(randomFloatVector(DIMENSION));

        // Create three identical cosine indices with different search paths
        createCosineIndex(STANDARD_INDEX, false, false);
        createCosineIndex(MEM_OPT_INDEX, true, false);
        createCosineIndex(EXACT_INDEX, false, true);

        // Ingest same data into all three
        bulkAddKnnDocs(STANDARD_INDEX, FIELD, vectors, NUM_DOCS);
        bulkAddKnnDocs(MEM_OPT_INDEX, FIELD, vectors, NUM_DOCS);
        bulkAddKnnDocs(EXACT_INDEX, FIELD, vectors, NUM_DOCS);

        forceMergeKnnIndex(STANDARD_INDEX, 1);
        forceMergeKnnIndex(MEM_OPT_INDEX, 1);
        forceMergeKnnIndex(EXACT_INDEX, 1);

        // Warmup memory optimized index
        knnWarmup(List.of(MEM_OPT_INDEX));

        // Search all three with the same query
        final Map<String, Float> standardScores = searchAndCollectScores(STANDARD_INDEX, query);
        final Map<String, Float> memOptScores = searchAndCollectScores(MEM_OPT_INDEX, query);
        final Map<String, Float> exactScores = searchAndCollectScores(EXACT_INDEX, query);

        // Verify all three return the same docs
        assertEquals("Standard vs MemOpt doc IDs should match", standardScores.keySet(), memOptScores.keySet());
        assertEquals("Standard vs Exact doc IDs should match", standardScores.keySet(), exactScores.keySet());

        // Verify scores are identical across all three paths
        for (Map.Entry<String, Float> entry : standardScores.entrySet()) {
            final String docId = entry.getKey();
            final float standardScore = entry.getValue();
            assertEquals("MemOpt score mismatch for doc " + docId, standardScore, memOptScores.get(docId), 1e-4f);
            assertEquals("Exact score mismatch for doc " + docId, standardScore, exactScores.get(docId), 1e-4f);
        }

        deleteKNNIndex(STANDARD_INDEX);
        deleteKNNIndex(MEM_OPT_INDEX);
        deleteKNNIndex(EXACT_INDEX);
    }

    private void createCosineIndex(String index, boolean memoryOptimized, boolean exactSearch) throws Exception {
        Settings.Builder builder = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(MEMORY_OPTIMIZED_KNN_SEARCH_MODE, memoryOptimized)
            .put(ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 1);

        if (exactSearch) {
            builder.put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, -1);
        }

        String mapping = createKnnIndexMapping(FIELD, DIMENSION, METHOD_HNSW, FAISS_NAME, SpaceType.COSINESIMIL.getValue());
        createKnnIndex(index, builder.build(), mapping);
    }

    private Map<String, Float> searchAndCollectScores(String index, float[] query) throws Exception {
        Response response = searchKNNIndex(index, KNNQueryBuilder.builder().k(K).fieldName(FIELD).vector(query).build(), K);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD);
        return results.stream().collect(Collectors.toMap(KNNResult::getDocId, KNNResult::getScore));
    }

    private float[][] generateNormalizedVectors(int count, int dim) {
        float[][] vectors = new float[count][];
        for (int i = 0; i < count; i++) {
            vectors[i] = normalizeVector(randomFloatVector(dim));
        }
        return vectors;
    }

    private float[] normalizeVector(float[] v) {
        float norm = 0;
        for (float f : v)
            norm += f * f;
        norm = (float) Math.sqrt(norm);
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++)
            out[i] = v[i] / norm;
        return out;
    }
}
