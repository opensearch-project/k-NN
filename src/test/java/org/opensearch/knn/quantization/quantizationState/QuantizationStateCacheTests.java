/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizationState;

import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateCache;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.opensearch.knn.quantization.enums.ScalarQuantizationType.ONE_BIT;

public class QuantizationStateCacheTests extends KNNTestCase {

    @SneakyThrows
    public void testSingleThreadedAddAndRetrieve() {
        String fieldName = "singleThreadField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );
        QuantizationStateCache cache = QuantizationStateCache.getInstance();

        // Add state
        cache.addQuantizationState(fieldName, state);

        QuantizationState retrievedState = cache.getQuantizationState(fieldName);
        assertNotNull("State should be retrieved successfully", retrievedState);
        assertSame("Retrieved state should be the same instance as the one added", state, retrievedState);
    }

    @SneakyThrows
    public void testMultiThreadedAddAndRetrieve() {
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        String fieldName = "multiThreadField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );
        QuantizationStateCache cache = QuantizationStateCache.getInstance();

        // Add state from multiple threads
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    cache.addQuantizationState(fieldName, state);
                } finally {
                    latch.countDown();
                }
            });
        }

        // Wait for all threads to finish
        latch.await();
        executorService.shutdown();

        QuantizationState retrievedState = cache.getQuantizationState(fieldName);
        assertNotNull("State should be retrieved successfully", retrievedState);
        assertSame("Retrieved state should be the same instance as the one added", state, retrievedState);
    }

    @SneakyThrows
    public void testMultiThreadedEvict() {
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        String fieldName = "multiThreadEvictField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );
        QuantizationStateCache cache = QuantizationStateCache.getInstance();

        cache.addQuantizationState(fieldName, state);

        // Evict state from multiple threads
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    cache.evict(fieldName);
                } finally {
                    latch.countDown();
                }
            });
        }

        // Wait for all threads to finish
        latch.await();
        executorService.shutdown();

        QuantizationState retrievedState = cache.getQuantizationState(fieldName);
        assertNull("State should be null", retrievedState);
    }

    @SneakyThrows
    public void testConcurrentAddAndEvict() {
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        String fieldName = "concurrentAddEvictField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );
        QuantizationStateCache cache = QuantizationStateCache.getInstance();

        // Concurrently add and evict state from multiple threads
        for (int i = 0; i < threadCount; i++) {
            if (i % 2 == 0) {
                executorService.submit(() -> {
                    try {
                        cache.addQuantizationState(fieldName, state);
                    } finally {
                        latch.countDown();
                    }
                });
            } else {
                executorService.submit(() -> {
                    try {
                        cache.evict(fieldName);
                    } finally {
                        latch.countDown();
                    }
                });
            }

        }

        // Wait for all threads to finish
        latch.await();
        executorService.shutdown();

        // Since operations are concurrent, we can't be sure of the final state, but we can assert that the cache handles it gracefully
        QuantizationState retrievedState = cache.getQuantizationState(fieldName);
        assertTrue("Final state should be either null or the added state", retrievedState == null || retrievedState == state);
    }

    @SneakyThrows
    public void testMultipleThreadedCacheClear() {
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        String fieldName = "multiThreadField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );
        QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.addQuantizationState(fieldName, state);

        // Clear cache from multiple threads
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    cache.clear();
                } finally {
                    latch.countDown();
                }
            });
        }

        // Wait for all threads to finish
        latch.await();
        executorService.shutdown();

        QuantizationState retrievedState = cache.getQuantizationState(fieldName);
        assertNull("State should be null", retrievedState);
    }
}
