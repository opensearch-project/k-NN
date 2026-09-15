/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.opensearch.common.collect.Tuple;
import org.opensearch.knn.KNNTestCase;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class KNN9120PerFieldKnnVectorsFormatTests extends KNNTestCase {

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsOne_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(1);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsZero_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(0);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyIsNegative_thenReturnsNullExecutor() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(-1);
        assertEquals(Integer.valueOf(1), result.v1());
        assertNull(result.v2());
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyAboveOne_thenReturnsExecutorWithCorrectPoolSize() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        try {
            assertEquals(Integer.valueOf(4), result.v1());
            assertNotNull(result.v2());
            assertTrue(result.v2() instanceof ThreadPoolExecutor);
            ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
            assertEquals(4, executor.getCorePoolSize());
            assertEquals(4, executor.getMaximumPoolSize());
        } finally {
            result.v2().shutdownNow();
        }
    }

    public void testBuildMergeThreadCountAndExecutorService_whenThreadQtyAboveOne_thenExecutorAllowsCoreThreadTimeout() {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(2);
        try {
            ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
            assertTrue(executor.allowsCoreThreadTimeOut());
        } finally {
            result.v2().shutdownNow();
        }
    }

    public void testBuildMergeThreadCountAndExecutorService_whenCalledMultipleTimes_thenReturnsIndependentExecutors() {
        Tuple<Integer, ExecutorService> first = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        Tuple<Integer, ExecutorService> second = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(4);
        try {
            assertNotSame(first.v2(), second.v2());
        } finally {
            first.v2().shutdownNow();
            second.v2().shutdownNow();
        }
    }

    /**
     * MrFlap test A: Verify that idle threads are culled after the keepAlive timeout expires.
     * Uses a 1-second timeout so the test completes quickly.
     */
    public void testThreadsCulledAfterTimeout() throws Exception {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(
            4,
            1L,
            TimeUnit.SECONDS
        );
        ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
        try {
            CountDownLatch tasksStarted = new CountDownLatch(4);
            CountDownLatch tasksCanFinish = new CountDownLatch(1);
            for (int i = 0; i < 4; i++) {
                executor.submit(() -> {
                    tasksStarted.countDown();
                    try {
                        tasksCanFinish.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            }
            assertTrue(tasksStarted.await(5, TimeUnit.SECONDS));
            assertEquals(4, executor.getActiveCount());

            tasksCanFinish.countDown();
            // Wait for keepAlive (1s) plus margin
            assertBusy(() -> assertEquals(0, executor.getPoolSize()), 10, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }
    }

    /**
     * MrFlap test B: Verify that threads remain alive while tasks are actively running,
     * even with a short keepAlive timeout.
     */
    public void testThreadsSurviveDuringActiveMerge() throws Exception {
        Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(
            4,
            1L,
            TimeUnit.SECONDS
        );
        ThreadPoolExecutor executor = (ThreadPoolExecutor) result.v2();
        try {
            CountDownLatch tasksStarted = new CountDownLatch(4);
            CountDownLatch tasksCanFinish = new CountDownLatch(1);
            for (int i = 0; i < 4; i++) {
                executor.submit(() -> {
                    tasksStarted.countDown();
                    try {
                        tasksCanFinish.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            }
            assertTrue(tasksStarted.await(5, TimeUnit.SECONDS));

            // Sleep past the keepAlive timeout — threads should still be alive because they're busy
            Thread.sleep(2000);
            assertEquals(4, executor.getPoolSize());
            assertEquals(4, executor.getActiveCount());

            tasksCanFinish.countDown();
        } finally {
            executor.shutdownNow();
        }
    }

    /**
     * MrFlap test C: Verify that executor objects become GC-eligible after use, so there
     * is no memory leak from accumulated executor instances.
     */
    public void testExecutorIsGarbageCollectedAfterUse() throws Exception {
        List<WeakReference<ExecutorService>> refs = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Tuple<Integer, ExecutorService> result = KNN9120PerFieldKnnVectorsFormat.buildMergeThreadCountAndExecutorService(
                2,
                1L,
                TimeUnit.SECONDS
            );
            refs.add(new WeakReference<>(result.v2()));
        }

        // Wait for threads to time out so nothing holds a strong reference to the executors
        Thread.sleep(3000);

        // Encourage GC
        for (int i = 0; i < 10; i++) {
            System.gc();
            Thread.sleep(100);
        }

        long collected = refs.stream().filter(ref -> ref.get() == null).count();
        // At least some executors should have been collected
        assertTrue("Expected at least one executor to be GC'd, but none were", collected > 0);
    }
}
