/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;
import org.junit.After;
import org.junit.Before;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.threadpool.Scheduler;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING;
import static org.opensearch.knn.index.KNNSettings.QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING;

public class QuantizationStateCacheTests extends KNNTestCase {

    private ThreadPool threadPool;

    @Before
    public void setThreadPool() {
        threadPool = new ThreadPool(Settings.builder().put("node.name", "QuantizationStateCacheTests").build());
        QuantizationStateCache.setThreadPool(threadPool);
        QuantizationStateCache.getInstance().rebuildCache();
    }

    @After
    public void terminateThreadPool() {
        terminate(threadPool);
    }

    @SneakyThrows
    public void testConcurrentLoadWhenValueExists() {
        // Set up thread executors
        final int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        final CountDownLatch latch = new CountDownLatch(threadCount);

        // Prepare quantization state
        final String fieldName = "multiThreadField";

        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
            .meanThresholds(new float[] { 1.2f, 2.3f, 3.4f })
            .build();

        // Configure settings
        final String cacheSize = "10%";
        final TimeValue expiry = TimeValue.timeValueMinutes(30);
        final Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Apply settings
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        clusterService.getClusterSettings().applySettings(settings);

        // Add the state first
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> state);
        assertEquals(state, retrievedState);

        // Add state from multiple threads
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    // Since we already added state at the beginning, even multiple threads try to load,
                    // the retrieved one should be the one that we added.
                    final QuantizationState acquired = cache.getQuantizationState(
                        fieldName,
                        () -> new OneBitScalarQuantizationState(
                            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
                            new float[] { 1.2f, 2.3f, 3.4f }
                        )
                    );
                    assertEquals(state, acquired);
                } finally {
                    latch.countDown();
                }
            });
        }

        // Wait for all threads to finish
        latch.await();
        executorService.shutdown();
    }

    @SneakyThrows
    public void testSingleThreadedAddAndRetrieve() {
        // Prepare state
        String fieldName = "singleThreadField";
        QuantizationState state = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            new float[] { 1.2f, 2.3f, 3.4f }
        );

        // Configure settings with 10%
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

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Apply the configured setting
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        clusterService.getClusterSettings().applySettings(settings);

        // Try to get a state and validate
        final QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> state);
        assertNotNull("State should be retrieved successfully", retrievedState);
        assertSame("Retrieved state should be the same instance as the one added", state, retrievedState);
    }

    @SneakyThrows
    public void testMultiThreadedAddAndRetrieve() {
        // Set up thread executors
        final int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        final CountDownLatch latch = new CountDownLatch(threadCount);

        // Prepare quantization state
        final String fieldName = "multiThreadField";
        final QuantizationState state = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            new float[] { 1.2f, 2.3f, 3.4f }
        );

        // Configure settings
        final String cacheSize = "10%";
        final TimeValue expiry = TimeValue.timeValueMinutes(30);
        final Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Apply settings
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        clusterService.getClusterSettings().applySettings(settings);

        // Add state from multiple threads
        final int tries = 100;
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    for (int k = 0; k < tries; k++) {
                        cache.getQuantizationState(fieldName, () -> state);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        // Wait for all threads to finish
        latch.await();
        executorService.shutdown();

        // Validate retrieved state
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> state);
        assertNotNull("State should be retrieved successfully", retrievedState);
        assertSame("Retrieved state should be the same instance as the one added", state, retrievedState);
    }

    @SneakyThrows
    public void testMultiThreadedEvict() {
        // Set up threads
        final int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);

        // Prepare quantization state
        String fieldName = "multiThreadEvictField";
        QuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
            .meanThresholds(new float[] { 1.2f, 2.3f, 3.4f })
            .build();
        String cacheSize = "10%";
        TimeValue expiry = TimeValue.timeValueMinutes(30);

        // Configure settings
        Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();

        // Mocking ClusterService
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Apply settings to ClusterService
        clusterService.getClusterSettings().applySettings(settings);

        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.getQuantizationState(fieldName, () -> state);

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

        final QuantizationState mockedState = getMockedState();
        final QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> mockedState);
        assertEquals(mockedState, retrievedState);
    }

    @SneakyThrows
    public void testConcurrentAddAndEvict() {
        // Set up thread executors
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);

        // Prepare quantization state
        final String fieldName = "concurrentAddEvictField";
        QuantizationState state = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            new float[] { 1.2f, 2.3f, 3.4f }
        );

        // Configure settings
        String cacheSize = "10%";
        TimeValue expiry = TimeValue.timeValueMinutes(30);
        Settings settings = Settings.builder()
            .put(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING.getKey(), cacheSize)
            .put(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING.getKey(), expiry)
            .build();

        // Mocking ClusterService
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Apply settings
        clusterService.getClusterSettings().applySettings(settings);

        // Concurrently add and evict state from multiple threads
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        for (int i = 0; i < threadCount; i++) {
            if (i % 2 == 0) {
                executorService.submit(() -> {
                    try {
                        cache.getQuantizationState(fieldName, () -> state);
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
        QuantizationState mockedState = getMockedState();
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> mockedState);
        assertTrue("Final state should be either new one or the added state", retrievedState == mockedState || retrievedState == state);
    }

    @SneakyThrows
    public void testMultipleThreadedCacheClear() {
        // Set up thread executors
        final int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);

        // Prepare quantization state
        final String fieldName = "multiThreadField";
        QuantizationState state = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            new float[] { 1.2f, 2.3f, 3.4f }
        );

        // Configure settings
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

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Apply settings
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        clusterService.getClusterSettings().applySettings(settings);
        cache.getQuantizationState(fieldName, () -> state);

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

        // Validate there's no state, and it should be the one we just added.
        QuantizationState mockedState = getMockedState();
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> mockedState);
        assertEquals(mockedState, retrievedState);
    }

    @SneakyThrows
    public void testRebuild() {
        // Set up thread executors
        final int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);

        // Prepare quantization state
        String fieldName = "rebuildField";
        QuantizationState state = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            new float[] { 1.2f, 2.3f, 3.4f }
        );

        // Configure settings
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

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Apply settings
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.getQuantizationState(fieldName, () -> state);

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

        // Validate there's no state, and it should be the one we just added.
        QuantizationState mockedState = getMockedState();
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> mockedState);
        assertEquals(mockedState, retrievedState);
    }

    @SneakyThrows
    public void testRebuildOnCacheSizeSettingsChange() {
        // Set up thread executors
        int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);

        // Prepare quantization state
        String fieldName = "rebuildField";
        QuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
            .meanThresholds(new float[] { 1.2f, 2.3f, 3.4f })
            .build();

        // Configure settings
        Settings settings = Settings.builder().build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Initialize KNNSettings
        Client client = mock(Client.class);
        KNNSettings.state().initialize(client, clusterService);

        // Rebuild and add the state
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        long maxCacheSizeInKB = cache.getMaxCacheSizeInKB();
        cache.getQuantizationState(fieldName, () -> state);

        // Prepare a new setting
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

        // Validate there's no state and KB threshold value.
        QuantizationState mockedState = getMockedState();
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> mockedState);
        assertEquals(mockedState, retrievedState);
        assertNotEquals(maxCacheSizeInKB, cache.getMaxCacheSizeInKB());
    }

    @SneakyThrows
    public void testRebuildOnTimeExpirySettingsChange() {
        // Set up thread executors
        final int threadCount = 10;
        ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);

        // Prepare quantization state
        String fieldName = "rebuildField";
        QuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
            .meanThresholds(new float[] { 1.2f, 2.3f, 3.4f })
            .build();

        // Configure settings
        Settings settings = Settings.builder().build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Initialize KNNSettings
        Client client = mock(Client.class);
        KNNSettings.state().initialize(client, clusterService);

        // Add a new state
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.getQuantizationState(fieldName, () -> state);

        // Prepare a new settings
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

        // Validate there was no state in it.
        QuantizationState mockedState = getMockedState();
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> mockedState);
        assertEquals(mockedState, retrievedState);
    }

    public void testCacheEvictionToSize() throws IOException {
        // Adding 4K + 100 bytes as meta info (e.g. length vint encoding etc)
        final int arrayLength = 1024;

        // Prepare state1 ~ roughly 4,100 bytes
        float[] meanThresholds1 = new float[arrayLength];
        for (int i = 0; i < arrayLength; i++) {
            meanThresholds1[i] = i;
        }

        // Configure settings
        Settings settings = Settings.builder().build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Build cache
        final String fieldName = "evictionField";
        // Setting 1KB as a threshold. As a result, expected the first one added will be evicted right away.
        final long cacheSizeKB = 1;
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.setMaxCacheSizeInKB(cacheSizeKB);
        cache.rebuildCache();  // Need to rebuild to update size threshold.

        // Try to add the first state
        final QuantizationState state = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            meanThresholds1
        );
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> state);
        assertEquals(state, retrievedState);

        // Try again
        final QuantizationState state2 = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            meanThresholds1
        );
        retrievedState = cache.getQuantizationState(fieldName, () -> state2);
        assertEquals(state2, retrievedState);

        // Close cache
        cache.clear();
        cache.close();

        // Validate whether states were evicted due to size.
        assertNotNull(cache.getEvictedDueToSizeAt());
    }

    public void testCacheEvictionDueToSize() throws IOException {
        // Adding 4K + 100 bytes as meta info (e.g. length vint encoding etc)
        final int arrayLength = 1024;

        // Prepare state1 ~ roughly 4,100 bytes
        float[] meanThresholds1 = new float[arrayLength];
        for (int i = 0; i < arrayLength; i++) {
            meanThresholds1[i] = i;
        }
        QuantizationState state1 = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            meanThresholds1
        );

        // Prepare state2 ~ roughly 4,100 bytes
        float[] meanThresholds2 = new float[arrayLength];
        for (int i = 0; i < arrayLength; i++) {
            meanThresholds2[i] = i + 1;
        }
        QuantizationState state2 = new OneBitScalarQuantizationState(
            ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build(),
            meanThresholds2
        );

        // Configure settings
        Settings settings = Settings.builder().build();
        ClusterSettings clusterSettings = new ClusterSettings(
            settings,
            ImmutableSet.of(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING)
        );

        // Mocking ClusterService
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(settings);

        // Build cache
        final String fieldName = "evictionField";
        final String fieldName2 = "evictionField2";
        // Setting 7KB as a threshold. As the weight of each one si roughly 4,100 bytes
        // Thus, setting 7KB so that it can evict the first one added when the second state is added.
        final long cacheSizeKB = 7;
        final QuantizationStateCache cache = QuantizationStateCache.getInstance();
        cache.setMaxCacheSizeInKB(cacheSizeKB);
        cache.rebuildCache();  // Need to rebuild to update size threshold.

        // Try to add the first state
        QuantizationState retrievedState = cache.getQuantizationState(fieldName, () -> state1);
        assertEquals(state1, retrievedState);

        // Try to add the second state
        retrievedState = cache.getQuantizationState(fieldName2, () -> state2);
        assertEquals(state2, retrievedState);

        // Close cache
        cache.clear();
        cache.close();

        // Validate whether states were evicted due to size.
        assertNotNull(cache.getEvictedDueToSizeAt());
    }

    public void testMaintenanceScheduled() throws Exception {
        final QuantizationStateCache quantizationStateCache = QuantizationStateCache.getInstance();
        Scheduler.Cancellable maintenanceTask = quantizationStateCache.getMaintenanceTask();

        assertNotNull(maintenanceTask);

        quantizationStateCache.close();
        assertTrue(maintenanceTask.isCancelled());
    }

    public void testMaintenanceWithRebuild() throws Exception {
        final QuantizationStateCache quantizationStateCache = QuantizationStateCache.getInstance();
        Scheduler.Cancellable task1 = quantizationStateCache.getMaintenanceTask();
        assertNotNull(task1);

        quantizationStateCache.rebuildCache();

        Scheduler.Cancellable task2 = quantizationStateCache.getMaintenanceTask();
        assertTrue(task1.isCancelled());
        assertNotNull(task2);
        quantizationStateCache.close();
    }

    @SneakyThrows
    private static QuantizationState getMockedState() {
        QuantizationState mockedState = mock(QuantizationState.class);
        when(mockedState.toByteArray()).thenReturn(new byte[32]);
        return mockedState;
    }
}
