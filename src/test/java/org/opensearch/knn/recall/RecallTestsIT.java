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

import com.google.common.primitives.Floats;
import org.apache.http.util.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.KNNQueryBuilder;
import org.opensearch.knn.index.SpaceType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY;
import static org.opensearch.knn.index.KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;

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
        float[][] queryVectors = TestUtils.getQueryVectors(queryCount, dimensions, true);
        float[][] groundTruthValues = TestUtils.computeGroundTruthDistances(indexVectors, queryVectors, SpaceType.L2, k);
        List<List<float[]>> searchResults = getSearchResults(testIndexStandard, testFieldName, queryVectors, k);
        double recallValue = TestUtils.calculateRecallValue(queryVectors, searchResults, groundTruthValues, k, SpaceType.L2);
        assertEquals(expRecallValue, recallValue, 0.2);
    }

    public void testRecallL2RandomData() throws Exception {
        String testIndexRandom = "test-index-random";

        addDocs(testIndexRandom, testFieldName, dimensions, docCount, false);
        float[][] indexVectors = getIndexVectorsFromIndex(testIndexRandom, testFieldName, docCount, dimensions);
        float[][] queryVectors = TestUtils.getQueryVectors(queryCount, dimensions, false);
        float[][] groundTruthValues = TestUtils.computeGroundTruthDistances(indexVectors, queryVectors, SpaceType.L2, k);
        List<List<float[]>> searchResults = getSearchResults(testIndexRandom, testFieldName, queryVectors, k);
        double recallValue = TestUtils.calculateRecallValue(queryVectors, searchResults, groundTruthValues, k, SpaceType.L2);
        assertEquals(expRecallValue, recallValue, 0.2);
    }

    private void addDocs(String testIndex, String testField, int dimensions, int docCount, boolean isStandard) throws Exception{
        createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(testField, dimensions));

        updateClusterSettings(KNN_ALGO_PARAM_INDEX_THREAD_QTY, 2);
        updateClusterSettings(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, true);

        bulkAddKnnDocs(testIndex, testField, TestUtils.getIndexVectors(docCount, dimensions, isStandard), docCount);
    }

    private List<List<float[]>> getSearchResults(String testIndex, String testField, float[][] queryVectors, int k) throws IOException {
        List<List<float[]>> searchResults = new ArrayList<>();
        List<float[]> kVectors;

        for(int i = 0; i < queryVectors.length; i++){
            KNNQueryBuilder knnQueryBuilderRecall = new KNNQueryBuilder(testField, queryVectors[i], k);
            Response respRecall = searchKNNIndex(testIndex, knnQueryBuilderRecall,k);
            List<KNNResult> resultsRecall = parseSearchResponse(EntityUtils.toString(respRecall.getEntity()), testField);

            assertEquals(resultsRecall.size(), k);
            kVectors = new ArrayList<>();
            for(KNNResult result : resultsRecall){
                kVectors.add(Floats.toArray(Arrays.stream(result.getVector()).collect(Collectors.toList())));
            }
            searchResults.add(kVectors);
        }

        return searchResults;
    }
}
