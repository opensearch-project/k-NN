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

import org.junit.BeforeClass;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.jni.JNIService;

import static org.mockito.Mockito.mockStatic;

public class SharedIndexStateManagerTests extends KNNTestCase {
    private static MockedStatic<JNIService> jniServiceMockedStatic;
    private final static long TEST_SHARED_TABLE_ADDRESS = 123;
    private final static long TEST_INDEX_ADDRESS = 1234;
    private final static String TEST_MODEL_ID = "test-model-id";
    private final static KNNEngine TEST_KNN_ENGINE = KNNEngine.DEFAULT;

    @BeforeClass
    public static void setUpClass() {
        jniServiceMockedStatic = mockStatic(JNIService.class);
        jniServiceMockedStatic.when(() -> JNIService.freeSharedIndexState(TEST_SHARED_TABLE_ADDRESS, TEST_KNN_ENGINE))
            .then(invocation -> null);
        jniServiceMockedStatic.when(() -> JNIService.initSharedIndexState(TEST_INDEX_ADDRESS, TEST_KNN_ENGINE))
            .thenReturn(TEST_SHARED_TABLE_ADDRESS);
    }

    public void testGet_whenNormalWorkfloatApplied_thenSucceed() {
        SharedIndexStateManager sharedIndexStateManager = new SharedIndexStateManager();
        SharedIndexState firstSharedIndexStateRetrieved = sharedIndexStateManager.get(TEST_INDEX_ADDRESS, TEST_MODEL_ID, TEST_KNN_ENGINE);
        assertEquals(TEST_SHARED_TABLE_ADDRESS, firstSharedIndexStateRetrieved.getSharedIndexStateAddress());
        assertEquals(TEST_MODEL_ID, firstSharedIndexStateRetrieved.getModelId());
        assertEquals(TEST_KNN_ENGINE, firstSharedIndexStateRetrieved.getKnnEngine());

        SharedIndexState secondSharedIndexStateRetrieved = sharedIndexStateManager.get(TEST_INDEX_ADDRESS, TEST_MODEL_ID, TEST_KNN_ENGINE);
        assertEquals(TEST_SHARED_TABLE_ADDRESS, secondSharedIndexStateRetrieved.getSharedIndexStateAddress());
        assertEquals(TEST_MODEL_ID, secondSharedIndexStateRetrieved.getModelId());
        assertEquals(TEST_KNN_ENGINE, secondSharedIndexStateRetrieved.getKnnEngine());
    }

    public void testRelease_whenNormalWorkflowApplied_thenSucceed() {
        SharedIndexStateManager sharedIndexStateManager = new SharedIndexStateManager();
        SharedIndexState firstSharedIndexStateRetrieved = sharedIndexStateManager.get(TEST_INDEX_ADDRESS, TEST_MODEL_ID, TEST_KNN_ENGINE);
        SharedIndexState secondSharedIndexStateRetrieved = sharedIndexStateManager.get(TEST_INDEX_ADDRESS, TEST_MODEL_ID, TEST_KNN_ENGINE);

        sharedIndexStateManager.release(firstSharedIndexStateRetrieved);
        jniServiceMockedStatic.verify(() -> JNIService.freeSharedIndexState(TEST_SHARED_TABLE_ADDRESS, TEST_KNN_ENGINE), Mockito.times(0));
        sharedIndexStateManager.release(secondSharedIndexStateRetrieved);
        jniServiceMockedStatic.verify(() -> JNIService.freeSharedIndexState(TEST_SHARED_TABLE_ADDRESS, TEST_KNN_ENGINE), Mockito.times(1));
    }
}
