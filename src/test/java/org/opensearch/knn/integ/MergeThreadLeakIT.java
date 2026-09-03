/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.List;
import java.util.Random;

import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY;

/**
 * Integration test that exercises the Lucene HNSW merge executor with
 * index_thread_qty > 1. Verifies that indexing, force-merging, and
 * searching all succeed — the merge executor must produce valid HNSW
 * graphs despite using a multi-threaded pool.
 *
 * This test targets the bug fixed in #3102 where each merge leaked a
 * FixedThreadPool that was never shut down.
 */
public class MergeThreadLeakIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "merge-thread-leak-test";
    private static final String FIELD_NAME = "test_vector";
    private static final int DIMENSION = 8;
    private static final int DOC_COUNT = 100;

    @SneakyThrows
    public void testLuceneHnswMerge_withMultipleThreads_thenSearchSucceeds() {
        updateClusterSettings(KNN_ALGO_PARAM_INDEX_THREAD_QTY, 4);
        try {
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(FIELD_NAME)
                .field("type", "knn_vector")
                .field("dimension", DIMENSION)
                .startObject("method")
                .field("name", "hnsw")
                .field("space_type", "l2")
                .field("engine", KNNEngine.LUCENE.getName())
                .startObject("parameters")
                .field("ef_construction", 16)
                .field("m", 4)
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .endObject()
                .toString();

            createKnnIndex(INDEX_NAME, mapping);

            Random rng = new Random(42);
            for (int i = 0; i < DOC_COUNT; i++) {
                Float[] vector = new Float[DIMENSION];
                for (int d = 0; d < DIMENSION; d++) {
                    vector[d] = rng.nextFloat();
                }
                addKnnDoc(INDEX_NAME, String.valueOf(i), FIELD_NAME, vector);
            }

            refreshAllNonSystemIndices();
            assertEquals(DOC_COUNT, getDocCount(INDEX_NAME));

            // Force merge exercises the per-field merge executor path that leaked threads
            forceMergeKnnIndex(INDEX_NAME);

            // A second force merge creates another round of executor allocations
            forceMergeKnnIndex(INDEX_NAME);

            Float[] queryVector = new Float[DIMENSION];
            for (int d = 0; d < DIMENSION; d++) {
                queryVector[d] = 0.5f;
            }
            String query = KNNJsonQueryBuilder.builder()
                .fieldName(FIELD_NAME)
                .vector(queryVector)
                .k(10)
                .build()
                .getQueryString();

            Response response = searchKNNIndex(INDEX_NAME, query, 10);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
            assertEquals(10, results.size());
        } finally {
            deleteKNNIndex(INDEX_NAME);
            updateClusterSettings(KNN_ALGO_PARAM_INDEX_THREAD_QTY, 1);
        }
    }
}
