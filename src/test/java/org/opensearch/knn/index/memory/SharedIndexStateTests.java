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

package org.opensearch.knn.index.memory;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

public class SharedIndexStateTests extends KNNTestCase {

    private static final String TEST_MODEL_ID = "test-model";
    private static final long TEST_SHARED_INDEX_STATE_ADDRESS = 22L;
    private static final KNNEngine TEST_KNN_ENGINE = KNNEngine.DEFAULT;

    public void testSharedIndexState() {
        SharedIndexState sharedIndexState = new SharedIndexState(TEST_SHARED_INDEX_STATE_ADDRESS, TEST_MODEL_ID, TEST_KNN_ENGINE);
        assertEquals(TEST_MODEL_ID, sharedIndexState.getModelId());
        assertEquals(TEST_SHARED_INDEX_STATE_ADDRESS, sharedIndexState.getSharedIndexStateAddress());
        assertEquals(TEST_KNN_ENGINE, sharedIndexState.getKnnEngine());
    }
}
