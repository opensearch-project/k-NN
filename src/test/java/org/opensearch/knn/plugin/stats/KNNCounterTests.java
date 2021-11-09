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

package org.opensearch.knn.plugin.stats;

import org.opensearch.knn.KNNTestCase;

public class KNNCounterTests extends KNNTestCase {
    public void testGetName() {
        assertEquals(StatNames.GRAPH_QUERY_ERRORS.getName(), KNNCounter.GRAPH_QUERY_ERRORS.getName());
    }

    public void testCount() {
        assertEquals((Long) 0L, KNNCounter.GRAPH_QUERY_ERRORS.getCount());

        for (long i = 0; i < 100; i++) {
            KNNCounter.GRAPH_QUERY_ERRORS.increment();
            assertEquals((Long) (i+1), KNNCounter.GRAPH_QUERY_ERRORS.getCount());
        }
    }
}