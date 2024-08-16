/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING;
import static org.opensearch.knn.index.KNNSettings.QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING;
import static org.opensearch.knn.quantization.enums.ScalarQuantizationType.ONE_BIT;

public class QuantizationStateCacheTests extends KNNTestCase {

    @SneakyThrows
    public void testSingleThreadedAddAndRetrieve() {
        String fieldName = "singleThreadField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );

        String cacheSize = "10%";
        TimeValue expiry = TimeValue.timeValueMinutes(30);

        Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        QuantizationStateCache cache = QuantizationStateCache.getInstance();
        clusterService.getClusterSettings().applySettings(settings);

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
        String cacheSize = "10%";
        TimeValue expiry = TimeValue.timeValueMinutes(30);

        Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        QuantizationStateCache cache = QuantizationStateCache.getInstance();
        clusterService.getClusterSettings().applySettings(settings);

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
        String cacheSize = "10%";
        TimeValue expiry = TimeValue.timeValueMinutes(30);

        Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        QuantizationStateCache cache = QuantizationStateCache.getInstance();

        clusterService.getClusterSettings().applySettings(settings);

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
        String cacheSize = "10%";
        TimeValue expiry = TimeValue.timeValueMinutes(30);

        Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        QuantizationStateCache cache = QuantizationStateCache.getInstance();
        clusterService.getClusterSettings().applySettings(settings);

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
        String cacheSize = "10%";
        TimeValue expiry = TimeValue.timeValueMinutes(30);

        Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        QuantizationStateCache cache = QuantizationStateCache.getInstance();
        clusterService.getClusterSettings().applySettings(settings);
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

    @SneakyThrows
    public void testRebuild() {
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        String fieldName = "rebuildField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );
        String cacheSize = "10%";
        TimeValue expiry = TimeValue.timeValueMinutes(30);

        Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.addQuantizationState(fieldName, state);

        // Rebuild cache from multiple threads
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    cache.rebuildCache();
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
    public void testRebuildOnCacheSizeSettingsChange() {
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        String fieldName = "rebuildField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );

        Settings settings = Settings.builder().build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        Client client = mock(Client.class);

        KNNSettings.state().initialize(client, clusterService);

        QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.rebuildCache();
        long maxCacheSizeInKB = cache.getMaxCacheSizeInKB();
        cache.addQuantizationState(fieldName, state);

        String newCacheSize = "10%";

        Settings newSettings = Settings.builder().put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), newCacheSize).build();

        // Rebuild cache from multiple threads
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    clusterService.getClusterSettings().applySettings(newSettings);
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
        assertNotEquals(maxCacheSizeInKB, cache.getMaxCacheSizeInKB());
    }

    @SneakyThrows
    public void testRebuildOnTimeExpirySettingsChange() {
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        String fieldName = "rebuildField";
        QuantizationState state = new OneBitScalarQuantizationState(
            new ScalarQuantizationParams(ONE_BIT),
            new float[] { 1.2f, 2.3f, 3.4f }
        );

        Settings settings = Settings.builder().build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        Client client = mock(Client.class);

        KNNSettings.state().initialize(client, clusterService);

        QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.addQuantizationState(fieldName, state);

        TimeValue newExpiry = TimeValue.timeValueMinutes(30);

        Settings newSettings = Settings.builder().put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), newExpiry).build();

        // Rebuild cache from multiple threads
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    clusterService.getClusterSettings().applySettings(newSettings);
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

    public void testCacheEvictionDueToSize() {
        String fieldName = "evictionField";
        // States have size of slightly over 500 bytes so that adding two will reach the max size of 1 kb for the cache
        int arrayLength = 112;
        float[] arr = new float[arrayLength];
        float[] arr2 = new float[arrayLength];
        for (int i = 0; i < arrayLength; i++) {
            arr[i] = i;
            arr[i] = i + 1;
        }
        QuantizationState state = new OneBitScalarQuantizationState(new ScalarQuantizationParams(ONE_BIT), arr);
        QuantizationState state2 = new OneBitScalarQuantizationState(new ScalarQuantizationParams(ONE_BIT), arr2);
        long cacheSize = 1;
        Settings settings = Settings.builder().build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        QuantizationStateCache cache = new QuantizationStateCache();
        cache.setMaxCacheSizeInKB(cacheSize);
        cache.rebuildCache();
        cache.addQuantizationState(fieldName, state);
        cache.addQuantizationState(fieldName, state2);
        cache.clear();
        assertNotNull(cache.getEvictedDueToSizeAt());
    }
}
