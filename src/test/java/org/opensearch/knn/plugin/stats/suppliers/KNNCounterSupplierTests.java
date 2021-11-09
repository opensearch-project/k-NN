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

package org.opensearch.knn.plugin.stats.suppliers;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.stats.KNNCounter;

public class KNNCounterSupplierTests extends KNNTestCase {
    public void testNormal() {
        KNNCounterSupplier knnCounterSupplier = new KNNCounterSupplier(KNNCounter.GRAPH_QUERY_REQUESTS);
        assertEquals((Long) 0L, knnCounterSupplier.get());
        KNNCounter.GRAPH_QUERY_REQUESTS.increment();
        assertEquals((Long) 1L, knnCounterSupplier.get());
    }
}