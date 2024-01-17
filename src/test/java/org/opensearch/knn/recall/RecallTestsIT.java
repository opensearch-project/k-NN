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

package org.opensearch.knn.recall;

import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.index.KNNCircuitBreaker.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;
import static org.opensearch.knn.jni.JNIService.KNN_ALGO_PARAM_INDEX_THREAD_QTY;

public class RecallTestsIT extends KNNRestTestCase {

    private final String testFieldName = "test-field";
    private final int dimensions = 50;
    private final int docCount = 10000;
    private final int queryCount = 100;
    private final int k = 5;
    private final double expRecallValue = 1.0;

    public void testRecallL2StandardData() throws Exception {
        String testIndexStandard = "test-index-standard";

        addDocs(testIndexStandard, testFieldName, dimensions, docCount, true);
        float[][] indexVectors = getIndexVectorsFromIndex(testIndexStandard, testFieldName, docCount, dimensions);
        float[][] queryVectors = TestUtils.getQueryVectors(queryCount, dimensions, docCount, true);
        List<Set<String>> groundTruthValues = TestUtils.computeGroundTruthValues(indexVectors, queryVectors, SpaceType.L2, k);
        List<List<String>> searchResults = bulkSearch(testIndexStandard, testFieldName, queryVectors, k);
        double recallValue = TestUtils.calculateRecallValue(searchResults, groundTruthValues, k);
        assertEquals(expRecallValue, recallValue, 0.2);
    }

    public void testRecallL2RandomData() throws Exception {
        String testIndexRandom = "test-index-random";

        addDocs(testIndexRandom, testFieldName, dimensions, docCount, false);
        float[][] indexVectors = getIndexVectorsFromIndex(testIndexRandom, testFieldName, docCount, dimensions);
        float[][] queryVectors = TestUtils.getQueryVectors(queryCount, dimensions, docCount, false);
        List<Set<String>> groundTruthValues = TestUtils.computeGroundTruthValues(indexVectors, queryVectors, SpaceType.L2, k);
        List<List<String>> searchResults = bulkSearch(testIndexRandom, testFieldName, queryVectors, k);
        double recallValue = TestUtils.calculateRecallValue(searchResults, groundTruthValues, k);
        assertEquals(expRecallValue, recallValue, 0.2);
    }

    private void addDocs(String testIndex, String testField, int dimensions, int docCount, boolean isStandard) throws Exception {
        createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(testField, dimensions));

        updateClusterSettings(KNN_ALGO_PARAM_INDEX_THREAD_QTY, 2);
        updateClusterSettings(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, true);

        bulkAddKnnDocs(testIndex, testField, TestUtils.getIndexVectors(docCount, dimensions, isStandard), docCount);
    }

}
