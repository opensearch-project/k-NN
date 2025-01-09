/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.rescore;

import org.opensearch.knn.KNNTestCase;

import static org.opensearch.knn.index.query.rescore.RescoreContext.MAX_FIRST_PASS_RESULTS;
import static org.opensearch.knn.index.query.rescore.RescoreContext.MIN_FIRST_PASS_RESULTS;

public class RescoreContextTests extends KNNTestCase {

    public void testGetFirstPassK() {
        float oversample = 2.6f;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(oversample).userProvided(true).build();
        int finalK = 100;
        boolean isShardLevelRescoringDisabled = false;
        int dimension = 500;

        // Case 1: Test with standard oversample factor when shard-level rescoring is enabled
        assertEquals(260, rescoreContext.getFirstPassK(finalK, isShardLevelRescoringDisabled, dimension));

        // Case 2: Test with a very small finalK that should result in a value less than MIN_FIRST_PASS_RESULTS
        finalK = 1;
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK, isShardLevelRescoringDisabled, dimension));

        // Case 3: Test with finalK = 0, should return MIN_FIRST_PASS_RESULTS
        finalK = 0;
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK, isShardLevelRescoringDisabled, dimension));

        // Case 4: Test with finalK = MAX_FIRST_PASS_RESULTS, should cap at MAX_FIRST_PASS_RESULTS
        finalK = MAX_FIRST_PASS_RESULTS;
        assertEquals(MAX_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK, isShardLevelRescoringDisabled, dimension));
    }

    public void testGetFirstPassKWithDimensionBasedOversampling() {
        int finalK = 100;
        int dimension;

        // Case 1: Test no oversampling for dimensions >= 1000 when shard-level rescoring is disabled
        dimension = 1000;
        RescoreContext rescoreContext = RescoreContext.builder().userProvided(false).build();  // Ensuring dimension-based logic applies
        assertEquals(100, rescoreContext.getFirstPassK(finalK, true, dimension));  // No oversampling

        // Case 2: Test 2x oversampling for dimensions >= 768 but < 1000 when shard-level rescoring is disabled
        dimension = 800;
        rescoreContext = RescoreContext.builder().userProvided(false).build();  // Ensure previous values don't carry over
        assertEquals(200, rescoreContext.getFirstPassK(finalK, true, dimension));  // 2x oversampling

        // Case 3: Test 3x oversampling for dimensions < 768 when shard-level rescoring is disabled
        dimension = 700;
        rescoreContext = RescoreContext.builder().userProvided(false).build();  // Ensure previous values don't carry over
        assertEquals(300, rescoreContext.getFirstPassK(finalK, true, dimension));  // 3x oversampling

        // Case 4: Shard-level rescoring enabled, oversample factor should be used as provided by the user (ignore dimension)
        rescoreContext = RescoreContext.builder().oversampleFactor(5.0f).userProvided(true).build();  // Provided by user
        dimension = 500;
        assertEquals(500, rescoreContext.getFirstPassK(finalK, false, dimension));  // User-defined oversample factor should be used

        // Case 5: Test finalK where oversampling factor results in a value less than MIN_FIRST_PASS_RESULTS
        finalK = 10;
        dimension = 700;
        rescoreContext = RescoreContext.builder().userProvided(false).build();  // Ensure dimension-based logic applies
        assertEquals(100, rescoreContext.getFirstPassK(finalK, true, dimension));  // 3x oversampling results in 30
    }

    public void testGetFirstPassKWithMinPassK() {
        float oversample = 0.5f;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(oversample).userProvided(true).build();  // User provided
        boolean isShardLevelRescoringDisabled = true;

        // Case 1: Test where finalK * oversample is smaller than MIN_FIRST_PASS_RESULTS
        int finalK = 10;
        int dimension = 700;
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK, isShardLevelRescoringDisabled, dimension));

        // Case 2: Test where finalK * oversample results in exactly MIN_FIRST_PASS_RESULTS
        finalK = 100;
        oversample = 1.0f;  // This will result in exactly 100 (MIN_FIRST_PASS_RESULTS)
        rescoreContext = RescoreContext.builder().oversampleFactor(oversample).userProvided(true).build();  // User provided
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK, isShardLevelRescoringDisabled, dimension));
    }
}
