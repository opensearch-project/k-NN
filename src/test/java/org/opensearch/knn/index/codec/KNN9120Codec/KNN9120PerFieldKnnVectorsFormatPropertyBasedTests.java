/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.opensearch.common.collect.Tuple;
import org.opensearch.knn.KNNTestCase;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Property-based tests for KNN9120PerFieldKnnVectorsFormat's getMergeThreadCountAndExecutorService method.
 */
public class KNN9120PerFieldKnnVectorsFormatPropertyBasedTests extends KNNTestCase {

    private static final int MIN_ITERATIONS = 100;

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
     * For any fixed thread count > 1, multiple calls return the same ExecutorService instance.
     */
    public void testExecutorInstanceReuse() throws Exception {
        for (int iteration = 0; iteration < MIN_ITERATIONS; iteration++) {
            int threadQty = randomIntBetween(2, 16);
            int callCount = randomIntBetween(2, 50);

            KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(threadQty);

            Tuple<Integer, ExecutorService> firstResult = invokeMergeThreadCountAndExecutorService();
            assertNotNull(firstResult.v2());

            for (int call = 1; call < callCount; call++) {
                Tuple<Integer, ExecutorService> result = invokeMergeThreadCountAndExecutorService();
                assertSame(firstResult.v2(), result.v2());
                assertEquals(Integer.valueOf(threadQty), result.v1());
            }
        }
    }

    /**
     * For any two distinct thread counts > 1, changing the count resizes the same executor
     * and the pool size matches the new value.
     */
    public void testSettingChangeResizesPool() throws Exception {
        for (int iteration = 0; iteration < MIN_ITERATIONS; iteration++) {
            int firstThreadQty = randomIntBetween(2, 16);
            int secondThreadQty = randomValueOtherThan(firstThreadQty, () -> randomIntBetween(2, 16));

            KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(firstThreadQty);
            Tuple<Integer, ExecutorService> firstResult = invokeMergeThreadCountAndExecutorService();
            assertNotNull(firstResult.v2());
            assertEquals(Integer.valueOf(firstThreadQty), firstResult.v1());

            KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(secondThreadQty);
            Tuple<Integer, ExecutorService> secondResult = invokeMergeThreadCountAndExecutorService();
            assertNotNull(secondResult.v2());
            assertEquals(Integer.valueOf(secondThreadQty), secondResult.v1());

            // Same executor instance, just resized
            assertSame(firstResult.v2(), secondResult.v2());
            assertFalse(secondResult.v2().isShutdown());
            assertEquals(secondThreadQty, ((ThreadPoolExecutor) secondResult.v2()).getCorePoolSize());
        }
    }

    /**
     * For any thread count > 1, the executor's getCorePoolSize() equals that value.
     */
    public void testPoolSizeMatchesSetting() throws Exception {
        for (int iteration = 0; iteration < MIN_ITERATIONS; iteration++) {
            int threadQty = randomIntBetween(2, 16);

            KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(threadQty);
            Tuple<Integer, ExecutorService> result = invokeMergeThreadCountAndExecutorService();

            assertNotNull(result.v2());
            assertEquals(Integer.valueOf(threadQty), result.v1());
            assertTrue(result.v2() instanceof ThreadPoolExecutor);
            assertEquals(threadQty, ((ThreadPoolExecutor) result.v2()).getCorePoolSize());
        }
    }

    /**
     * For any concurrency level, all threads receive the same non-null, non-shutdown executor.
     */
    public void testConcurrentAccessSafety() throws Exception {
        for (int iteration = 0; iteration < MIN_ITERATIONS; iteration++) {
            int threadQty = randomIntBetween(2, 16);
            int concurrencyLevel = randomIntBetween(2, 20);

            KNN9120PerFieldKnnVectorsFormat.updateMergeThreadCount(threadQty);

            Tuple<Integer, ExecutorService> primed = invokeMergeThreadCountAndExecutorService();
            assertNotNull(primed.v2());

            CyclicBarrier barrier = new CyclicBarrier(concurrencyLevel);
            ExecutorService testExecutor = Executors.newFixedThreadPool(concurrencyLevel);

            try {
                List<Future<Tuple<Integer, ExecutorService>>> futures = new ArrayList<>(concurrencyLevel);
                for (int t = 0; t < concurrencyLevel; t++) {
                    futures.add(testExecutor.submit(() -> {
                        barrier.await();
                        return invokeMergeThreadCountAndExecutorService();
                    }));
                }

                for (Future<Tuple<Integer, ExecutorService>> future : futures) {
                    Tuple<Integer, ExecutorService> result = future.get();
                    assertNotNull(result.v2());
                    assertFalse(result.v2().isShutdown());
                    assertSame(primed.v2(), result.v2());
                    assertEquals(Integer.valueOf(threadQty), result.v1());
                }
            } finally {
                testExecutor.shutdown();
            }
        }
    }
}
