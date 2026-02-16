/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.common.collect.Tuple;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Unit tests for boundary conditions in KNN9120PerFieldKnnVectorsFormat's
 * getMergeThreadCountAndExecutorService method.
 */
public class KNN9120PerFieldKnnVectorsFormatTests extends KNNTestCase {

    @Override
    public void setUp() throws Exception {
        super.setUp();
        resetCachedExecutorState();
    }

    @Override
    public void tearDown() throws Exception {
        resetCachedExecutorState();
        super.tearDown();
    }

    /**
     * Reset the static cached executor fields via reflection so tests are isolated.
     */
    private void resetCachedExecutorState() throws Exception {
        Field executorField = KNN9120PerFieldKnnVectorsFormat.class.getDeclaredField("cachedMergeExecutorService");
        executorField.setAccessible(true);
        ExecutorService existing = (ExecutorService) executorField.get(null);
        if (existing != null) {
            existing.shutdown();
        }
        executorField.set(null, null);

        Field threadCountField = KNN9120PerFieldKnnVectorsFormat.class.getDeclaredField("cachedMergeThreadCount");
        threadCountField.setAccessible(true);
        threadCountField.set(null, 0);
    }

    /**
     * Invoke the private static synchronized getMergeThreadCountAndExecutorService method via reflection.
     */
    @SuppressWarnings("unchecked")
    private Tuple<Integer, ExecutorService> invokeMergeThreadCountAndExecutorService() throws Exception {
        Method method = KNN9120PerFieldKnnVectorsFormat.class.getDeclaredMethod("getMergeThreadCountAndExecutorService");
        method.setAccessible(true);
        return (Tuple<Integer, ExecutorService>) method.invoke(null);
    }

    /**
     * When Index_Thread_Qty = 1, the method should return (1, null).
     */
    public void testGetMergeThreadCount_whenThreadQtyIsOne_thenReturnsDefaultTuple() throws Exception {
        try (MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)) {
            mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(1);
            mockedSettings.when(KNNSettings::state).thenCallRealMethod();

            Tuple<Integer, ExecutorService> result = invokeMergeThreadCountAndExecutorService();

            assertEquals(Integer.valueOf(1), result.v1());
            assertNull(result.v2());
        }
    }

    /**
     * When Index_Thread_Qty = 0, the method should return (1, null).
     */
    public void testGetMergeThreadCount_whenThreadQtyIsZero_thenReturnsDefaultTuple() throws Exception {
        try (MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)) {
            mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(0);
            mockedSettings.when(KNNSettings::state).thenCallRealMethod();

            Tuple<Integer, ExecutorService> result = invokeMergeThreadCountAndExecutorService();

            assertEquals(Integer.valueOf(1), result.v1());
            assertNull(result.v2());
        }
    }

    /**
     * When Index_Thread_Qty transitions from > 1 to <= 1, the cached executor should be shut down.
     */
    public void testGetMergeThreadCount_whenTransitionFromHighToLow_thenShutsCachedExecutor() throws Exception {
        try (MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)) {
            mockedSettings.when(KNNSettings::state).thenCallRealMethod();

            // First call with thread qty > 1 to create a cached executor
            mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(4);
            Tuple<Integer, ExecutorService> highResult = invokeMergeThreadCountAndExecutorService();
            ExecutorService createdExecutor = highResult.v2();
            assertNotNull(createdExecutor);
            assertFalse(createdExecutor.isShutdown());

            // Now transition to thread qty <= 1
            mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(1);
            Tuple<Integer, ExecutorService> lowResult = invokeMergeThreadCountAndExecutorService();

            assertEquals(Integer.valueOf(1), lowResult.v1());
            assertNull(lowResult.v2());
            // The previously cached executor should now be shut down
            assertTrue(createdExecutor.isShutdown());
        }
    }

    /**
     * When Index_Thread_Qty transitions from <= 1 to > 1, a new executor should be created.
     */
    public void testGetMergeThreadCount_whenTransitionFromLowToHigh_thenCreatesNewExecutor() throws Exception {
        try (MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)) {
            mockedSettings.when(KNNSettings::state).thenCallRealMethod();

            // First call with thread qty <= 1
            mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(1);
            Tuple<Integer, ExecutorService> lowResult = invokeMergeThreadCountAndExecutorService();
            assertNull(lowResult.v2());

            // Now transition to thread qty > 1
            mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(4);
            Tuple<Integer, ExecutorService> highResult = invokeMergeThreadCountAndExecutorService();

            assertEquals(Integer.valueOf(4), highResult.v1());
            assertNotNull(highResult.v2());
            assertFalse(highResult.v2().isShutdown());
            assertTrue(highResult.v2() instanceof ThreadPoolExecutor);
            assertEquals(4, ((ThreadPoolExecutor) highResult.v2()).getCorePoolSize());
        }
    }
}
