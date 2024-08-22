/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.rescore;

import org.opensearch.knn.KNNTestCase;

import static org.opensearch.knn.index.query.rescore.RescoreContext.MAX_FIRST_PASS_RESULTS;

public class RescoreContextTests extends KNNTestCase {

    public void testGetFirstPassK() {
        float oversample = 2.6f;
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(oversample).build();
        int finalK = 100;
        assertEquals(260, rescoreContext.getFirstPassK(finalK));
        finalK = 1;
        assertEquals(3, rescoreContext.getFirstPassK(finalK));
        finalK = 0;
        assertEquals(0, rescoreContext.getFirstPassK(finalK));
        finalK = MAX_FIRST_PASS_RESULTS;
        assertEquals(MAX_FIRST_PASS_RESULTS, rescoreContext.getFirstPassK(finalK));
    }
}
