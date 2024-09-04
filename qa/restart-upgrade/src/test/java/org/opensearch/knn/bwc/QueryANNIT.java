/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;

/**
 * Use case: Test queries on indexes created on older versions
 */
public class QueryANNIT extends AbstractRestartUpgradeTestCase {

    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final Integer EF_SEARCH = 10;
    private static final int NUM_DOCS = 10;
    private static final String ALGORITHM = "hnsw";

    public void testQueryOnFaissIndex() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGORITHM, FAISS_NAME));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K, Map.of(METHOD_PARAMETER_EF_SEARCH, EF_SEARCH));
            deleteKNNIndex(testIndex);
        }
    }

    public void testQueryOnNmslibIndex() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGORITHM, NMSLIB_NAME));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K, Map.of(METHOD_PARAMETER_EF_SEARCH, EF_SEARCH));
            deleteKNNIndex(testIndex);
        }
    }

    public void testQueryOnLuceneIndex() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, ALGORITHM, LUCENE_NAME));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K, Map.of(METHOD_PARAMETER_EF_SEARCH, EF_SEARCH));
            deleteKNNIndex(testIndex);
        }
    }
}
