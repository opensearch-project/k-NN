/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.opensearch.common.collect.Tuple;
import org.opensearch.knn.KNNTestCase;

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
        KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(0);
    }

    @Override
    public void tearDown() throws Exception {
        KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(0);
        super.tearDown();
    }

    @SuppressWarnings("unchecked")
    private Tuple<Integer, ExecutorService> invokeMergeThreadCountAndExecutorService() throws Exception {
        Method method = KNN9120PerFieldKnnVectorsFormat.class.getDeclaredMethod("getMergeThreadCountAndExecutorService");
        method.setAccessible(true);
        return (Tuple<Integer, ExecutorService>) method.invoke(null);
    }

    /**
     * When thread count is 1, the method should return (1, null).
     */
    public void testGetMergeThreadCount_whenThreadQtyIsOne_thenReturnsDefaultTuple() throws Exception {
        KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(1);

        Tuple<Integer, ExecutorService> result = invokeMergeThreadCountAndExecutorService();

        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    /**
     * When thread count is 0, the method should return (1, null).
     */
    public void testGetMergeThreadCount_whenThreadQtyIsZero_thenReturnsDefaultTuple() throws Exception {
        KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(0);

        Tuple<Integer, ExecutorService> result = invokeMergeThreadCountAndExecutorService();

        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    /**
     * When thread count transitions from > 1 to <= 1, the method returns null executor.
     */
    public void testGetMergeThreadCount_whenTransitionFromHighToLow_thenReturnsNull() throws Exception {
        KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(4);
        Tuple<Integer, ExecutorService> highResult = invokeMergeThreadCountAndExecutorService();
        assertNotNull(highResult.v2());

        KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(1);
        Tuple<Integer, ExecutorService> lowResult = invokeMergeThreadCountAndExecutorService();

        assertEquals(Integer.valueOf(1), lowResult.v1());
        assertNull(lowResult.v2());
    }

    /**
     * When thread count transitions from <= 1 to > 1, the executor is returned with correct pool size.
     */
    public void testGetMergeThreadCount_whenTransitionFromLowToHigh_thenReturnsExecutor() throws Exception {
        KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(1);
        Tuple<Integer, ExecutorService> lowResult = invokeMergeThreadCountAndExecutorService();
        assertNull(lowResult.v2());

        KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(4);
        Tuple<Integer, ExecutorService> highResult = invokeMergeThreadCountAndExecutorService();

        assertEquals(Integer.valueOf(4), highResult.v1());
        assertNotNull(highResult.v2());
        assertFalse(highResult.v2().isShutdown());
        assertTrue(highResult.v2() instanceof ThreadPoolExecutor);
        assertEquals(4, ((ThreadPoolExecutor) highResult.v2()).getCorePoolSize());
    }
}
