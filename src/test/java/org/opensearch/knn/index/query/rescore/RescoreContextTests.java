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
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(oversample).build();
        int finalK = 100;
        assertEquals(260, rescoreContext.getFirstPassK(finalK));
        finalK = 1;
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));
        finalK = 0;
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));
        finalK = MAX_FIRST_PASS_RESULTS;
        assertEquals(MAX_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));
    }

    public void testGetFirstPassKWithMinPassK() {
        float oversample = 2.6f;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(oversample).build();

        // Case 1: Test with a finalK that results in a value greater than MIN_FIRST_PASS_RESULTS
        int finalK = 100;
        assertEquals(260, rescoreContext.getFirstPassK(finalK));

        // Case 2: Test with a very small finalK that should result in a value less than MIN_FIRST_PASS_RESULTS
        finalK = 1;
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));

        // Case 3: Test with finalK = 0, should return 0
        finalK = 0;
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));

        // Case 4: Test with finalK = MAX_FIRST_PASS_RESULTS, should cap at MAX_FIRST_PASS_RESULTS
        finalK = MAX_FIRST_PASS_RESULTS;
        assertEquals(MAX_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));

        // Case 5: Test where finalK * oversample is smaller than MIN_FIRST_PASS_RESULTS
        finalK = 10;
        oversample = 0.5f;  // This will result in 5, which is less than MIN_FIRST_PASS_RESULTS
        rescoreContext = RescoreContext.builder().oversampleFactor(oversample).build();
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));

        // Case 6: Test where finalK * oversample results in exactly MIN_FIRST_PASS_RESULTS
        finalK = 100;
        oversample = 1.0f;  // This will result in exactly 100 (MIN_FIRST_PASS_RESULTS)
        rescoreContext = RescoreContext.builder().oversampleFactor(oversample).build();
        assertEquals(MIN_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));
    }
}
