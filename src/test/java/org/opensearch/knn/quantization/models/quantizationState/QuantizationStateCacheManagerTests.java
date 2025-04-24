/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.SneakyThrows;
import org.junit.After;
import org.junit.Before;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN990Codec.KNN990QuantizationStateReader;
import org.opensearch.threadpool.ThreadPool;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.when;

public class QuantizationStateCacheManagerTests extends KNNTestCase {

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
    public void testConcurrentLoad() {
        // Get manager and clean it.
        final QuantizationStateCacheManager manager = QuantizationStateCacheManager.getInstance();
        manager.rebuildCache();

        // Mock read config
        final QuantizationStateReadConfig readConfig = mock(QuantizationStateReadConfig.class);
        when(readConfig.getCacheKey()).thenReturn("cache_key");

        // Add state first.
        final QuantizationState quantizationState = mock(QuantizationState.class);
        when(quantizationState.toByteArray()).thenReturn(new byte[32]);
        try (MockedStatic<KNN990QuantizationStateReader> mockedStaticReader = Mockito.mockStatic(KNN990QuantizationStateReader.class)) {
            // Mock static
            mockedStaticReader.when(() -> KNN990QuantizationStateReader.read(readConfig)).thenReturn(quantizationState);

            // Add state
            manager.getQuantizationState(readConfig);
        }

        // Set up thread executors
        final int threadCount = 10;
        final int tries = 100;
        final ExecutorService executorService = Executors.newFixedThreadPool(threadCount);
        final CountDownLatch latch = new CountDownLatch(threadCount);

        // Try to get in parallel
        for (int i = 0; i < threadCount; i++) {
            executorService.submit(() -> {
                try {
                    for (int k = 0; k < tries; k++) {
                        // Since we already added state at the beginning, even multiple threads try to load,
                        // the retrieved one should be the one that we added.
                        final QuantizationState acquired = manager.getQuantizationState(readConfig);
                        assertEquals(quantizationState, acquired);
                    }
                } catch (Exception e) {
                    fail(e.getMessage());
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
    public void testRebuildCache() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            // Mocking state cache singleton
            QuantizationStateCache quantizationStateCache = mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);

            // Mocking it to do nothing when `rebuildCache`
            Mockito.doNothing().when(quantizationStateCache).rebuildCache();
            QuantizationStateCacheManager.getInstance().rebuildCache();

            // Verify rebuildCache is called exactly once
            Mockito.verify(quantizationStateCache, times(1)).rebuildCache();
        }
    }

    @SneakyThrows
    public void testGetQuantizationState() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            // Mocking read config with cache key
            QuantizationStateReadConfig quantizationStateReadConfig = mock(QuantizationStateReadConfig.class);
            String cacheKey = "test-key";
            when(quantizationStateReadConfig.getCacheKey()).thenReturn(cacheKey);

            // Mocking quantization state
            QuantizationState quantizationState = mock(QuantizationState.class);
            QuantizationStateCache quantizationStateCache = mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            when(quantizationStateCache.getQuantizationState(any(), any())).thenReturn(quantizationState);

            // Validate `getQuantizationState` of `quantizationStateCache` was called.
            try (MockedStatic<KNN990QuantizationStateReader> mockedStaticReader = Mockito.mockStatic(KNN990QuantizationStateReader.class)) {
                mockedStaticReader.when(() -> KNN990QuantizationStateReader.read(quantizationStateReadConfig))
                    .thenReturn(quantizationState);
                QuantizationStateCacheManager.getInstance().getQuantizationState(quantizationStateReadConfig);
                Mockito.verify(quantizationStateCache, times(1)).getQuantizationState(eq(cacheKey), any());
            }

            // Validate `getQuantizationState` was called AGAIN.
            // But this time, we don't need to invoke `read` as we have a value loaded already.
            when(quantizationStateCache.getQuantizationState(any(), any())).thenReturn(quantizationState);
            QuantizationStateCacheManager.getInstance().getQuantizationState(quantizationStateReadConfig);
            Mockito.verify(quantizationStateCache, times(2)).getQuantizationState(eq(cacheKey), any());
        }
    }

    @SneakyThrows
    public void testEvict() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            String field = "test-field";
            QuantizationStateCache quantizationStateCache = mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).evict(field);
            QuantizationStateCacheManager.getInstance().evict(field);
            Mockito.verify(quantizationStateCache, times(1)).evict(field);
        }
    }

    @SneakyThrows
    public void testSetMaxCacheSizeInKB() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            long maxCacheSizeInKB = 1024;
            QuantizationStateCache quantizationStateCache = mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).setMaxCacheSizeInKB(maxCacheSizeInKB);
            QuantizationStateCacheManager.getInstance().setMaxCacheSizeInKB(1024);
            Mockito.verify(quantizationStateCache, times(1)).setMaxCacheSizeInKB(1024);
        }
    }

    @SneakyThrows
    public void testClear() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            QuantizationStateCache quantizationStateCache = mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).clear();
            QuantizationStateCacheManager.getInstance().clear();
            Mockito.verify(quantizationStateCache, times(1)).clear();
        }
    }
}
