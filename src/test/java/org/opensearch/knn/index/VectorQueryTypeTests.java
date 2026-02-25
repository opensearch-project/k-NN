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

package org.opensearch.knn.index;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.stats.KNNCounter;

public class VectorQueryTypeTests extends KNNTestCase {

    public void testGetQueryStatCounter() {
        assertEquals(KNNCounter.KNN_QUERY_REQUESTS, VectorQueryType.K.getQueryStatCounter());
        assertEquals(KNNCounter.MIN_SCORE_QUERY_REQUESTS, VectorQueryType.MIN_SCORE.getQueryStatCounter());
        assertEquals(KNNCounter.MAX_DISTANCE_QUERY_REQUESTS, VectorQueryType.MAX_DISTANCE.getQueryStatCounter());
    }

    public void testGetQueryWithFilterStatCounter() {
        assertEquals(KNNCounter.KNN_QUERY_WITH_FILTER_REQUESTS, VectorQueryType.K.getQueryWithFilterStatCounter());
        assertEquals(KNNCounter.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS, VectorQueryType.MIN_SCORE.getQueryWithFilterStatCounter());
        assertEquals(KNNCounter.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS, VectorQueryType.MAX_DISTANCE.getQueryWithFilterStatCounter());
    }
}
