/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

public class FaissEngineTests extends KNNTestCase {
    public void testNonFaissEngineToReturnNullSearcher() {
        assertNull(KNNEngine.LUCENE.getVectorSearcherFactory());
        assertNull(KNNEngine.NMSLIB.getVectorSearcherFactory());
    }
}
