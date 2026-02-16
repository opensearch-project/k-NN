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
     * For any fixed Index_Thread_Qty > 1, multiple calls to getMergeThreadCountAndExecutorService()
     * return the same ExecutorService instance (reference equality).
     */
    public void testExecutorInstanceReuse() throws Exception {
        for (int iteration = 0; iteration < MIN_ITERATIONS; iteration++) {
            resetCachedExecutorState();

            int threadQty = randomIntBetween(2, 16);
            int callCount = randomIntBetween(2, 50);

            try (MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)) {
                mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(threadQty);
                mockedSettings.when(KNNSettings::state).thenCallRealMethod();

                // First call establishes the cached executor
                Tuple<Integer, ExecutorService> firstResult = invokeMergeThreadCountAndExecutorService();
                assertNotNull(
                    "Iteration " + iteration + ": first call with threadQty=" + threadQty + " should return non-null executor",
                    firstResult.v2()
                );

                // Subsequent calls should return the exact same instance
                for (int call = 1; call < callCount; call++) {
                    Tuple<Integer, ExecutorService> result = invokeMergeThreadCountAndExecutorService();
                    assertSame(
                        "Iteration "
                            + iteration
                            + ": call "
                            + call
                            + " of "
                            + callCount
                            + " with threadQty="
                            + threadQty
                            + " should return same executor instance (reference equality)",
                        firstResult.v2(),
                        result.v2()
                    );
                    assertEquals("Iteration " + iteration + ": thread count should match setting", Integer.valueOf(threadQty), result.v1());
                }
            }
        }
    }

    /**
     * For any two distinct values of Index_Thread_Qty both greater than 1, changing the setting
     * from the first value to the second and calling getMergeThreadCountAndExecutorService()
     * returns a different ExecutorService instance, and the previous instance is shut down.
     */
    public void testSettingChangeCausesRecreation() throws Exception {
        for (int iteration = 0; iteration < MIN_ITERATIONS; iteration++) {
            resetCachedExecutorState();

            int firstThreadQty = randomIntBetween(2, 16);
            int secondThreadQty = randomValueOtherThan(firstThreadQty, () -> randomIntBetween(2, 16));

            try (MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)) {
                mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(firstThreadQty);
                mockedSettings.when(KNNSettings::state).thenCallRealMethod();

                // First call with the initial thread count
                Tuple<Integer, ExecutorService> firstResult = invokeMergeThreadCountAndExecutorService();
                assertNotNull(
                    "Iteration " + iteration + ": first call with threadQty=" + firstThreadQty + " should return non-null executor",
                    firstResult.v2()
                );
                assertEquals(
                    "Iteration " + iteration + ": first thread count should match setting",
                    Integer.valueOf(firstThreadQty),
                    firstResult.v1()
                );

                ExecutorService firstExecutor = firstResult.v2();

                // Change the setting to a different value
                mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(secondThreadQty);

                // Second call should produce a new executor
                Tuple<Integer, ExecutorService> secondResult = invokeMergeThreadCountAndExecutorService();
                assertNotNull(
                    "Iteration " + iteration + ": second call with threadQty=" + secondThreadQty + " should return non-null executor",
                    secondResult.v2()
                );
                assertEquals(
                    "Iteration " + iteration + ": second thread count should match new setting",
                    Integer.valueOf(secondThreadQty),
                    secondResult.v1()
                );

                assertNotSame(
                    "Iteration "
                        + iteration
                        + ": changing threadQty from "
                        + firstThreadQty
                        + " to "
                        + secondThreadQty
                        + " should produce a different executor instance",
                    firstExecutor,
                    secondResult.v2()
                );

                assertTrue(
                    "Iteration "
                        + iteration
                        + ": old executor should be shut down after setting change from "
                        + firstThreadQty
                        + " to "
                        + secondThreadQty,
                    firstExecutor.isShutdown()
                );
            }
        }
    }

    /**
     * For any value of Index_Thread_Qty greater than 1, the ExecutorService returned by
     * getMergeThreadCountAndExecutorService() is a ThreadPoolExecutor with getCorePoolSize()
     * equal to that value.
     */
    public void testPoolSizeMatchesSetting() throws Exception {
        for (int iteration = 0; iteration < MIN_ITERATIONS; iteration++) {
            resetCachedExecutorState();

            int threadQty = randomIntBetween(2, 16);

            try (MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)) {
                mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(threadQty);
                mockedSettings.when(KNNSettings::state).thenCallRealMethod();

                Tuple<Integer, ExecutorService> result = invokeMergeThreadCountAndExecutorService();
                assertNotNull("Iteration " + iteration + ": executor should be non-null for threadQty=" + threadQty, result.v2());
                assertEquals(
                    "Iteration " + iteration + ": returned thread count should match setting",
                    Integer.valueOf(threadQty),
                    result.v1()
                );

                assertTrue(
                    "Iteration " + iteration + ": executor should be a ThreadPoolExecutor for threadQty=" + threadQty,
                    result.v2() instanceof ThreadPoolExecutor
                );

                ThreadPoolExecutor poolExecutor = (ThreadPoolExecutor) result.v2();
                assertEquals(
                    "Iteration " + iteration + ": core pool size should equal threadQty=" + threadQty,
                    threadQty,
                    poolExecutor.getCorePoolSize()
                );
            }
        }
    }

    /**
     * For any number of concurrent threads calling getMergeThreadCountAndExecutorService()
     * simultaneously (with a fixed Index_Thread_Qty > 1), every thread receives a non-null,
     * non-shutdown ExecutorService, and all threads receive the same instance.
     */
    public void testConcurrentAccessSafety() throws Exception {
        for (int iteration = 0; iteration < MIN_ITERATIONS; iteration++) {
            resetCachedExecutorState();

            int threadQty = randomIntBetween(2, 16);
            int concurrencyLevel = randomIntBetween(2, 20);

            try (MockedStatic<KNNSettings> mockedSettings = Mockito.mockStatic(KNNSettings.class)) {
                mockedSettings.when(KNNSettings::getIndexThreadQty).thenReturn(threadQty);
                mockedSettings.when(KNNSettings::state).thenCallRealMethod();

                // Prime the cache in the main thread where MockedStatic is active.
                Tuple<Integer, ExecutorService> primed = invokeMergeThreadCountAndExecutorService();
                assertNotNull("Iteration " + iteration + ": primed executor should be non-null for threadQty=" + threadQty, primed.v2());

                // Spawn concurrent threads. Each thread creates its own MockedStatic scope
                // so that KNNSettings.getIndexThreadQty() returns the same threadQty value,
                // ensuring the synchronized method hits the cached path.
                CyclicBarrier barrier = new CyclicBarrier(concurrencyLevel);
                ExecutorService testExecutor = Executors.newFixedThreadPool(concurrencyLevel);
                final int capturedThreadQty = threadQty;

                try {
                    List<Future<Tuple<Integer, ExecutorService>>> futures = new ArrayList<>(concurrencyLevel);
                    for (int t = 0; t < concurrencyLevel; t++) {
                        futures.add(testExecutor.submit(() -> {
                            try (MockedStatic<KNNSettings> threadMock = Mockito.mockStatic(KNNSettings.class)) {
                                threadMock.when(KNNSettings::getIndexThreadQty).thenReturn(capturedThreadQty);
                                threadMock.when(KNNSettings::state).thenCallRealMethod();
                                barrier.await();
                                return invokeMergeThreadCountAndExecutorService();
                            }
                        }));
                    }

                    List<Tuple<Integer, ExecutorService>> results = new ArrayList<>(concurrencyLevel);
                    for (Future<Tuple<Integer, ExecutorService>> future : futures) {
                        results.add(future.get());
                    }

                    for (int i = 0; i < results.size(); i++) {
                        Tuple<Integer, ExecutorService> result = results.get(i);
                        assertNotNull(
                            "Iteration " + iteration + ": concurrent thread " + i + " should have non-null executor",
                            result.v2()
                        );
                        assertFalse(
                            "Iteration " + iteration + ": concurrent thread " + i + " executor should not be shut down",
                            result.v2().isShutdown()
                        );
                        assertSame(
                            "Iteration " + iteration + ": concurrent thread " + i + " should receive same executor instance as primed",
                            primed.v2(),
                            result.v2()
                        );
                        assertEquals(
                            "Iteration " + iteration + ": concurrent thread " + i + " thread count should match setting",
                            Integer.valueOf(capturedThreadQty),
                            result.v1()
                        );
                    }
                } finally {
                    testExecutor.shutdown();
                }
            }
        }
    }

}
