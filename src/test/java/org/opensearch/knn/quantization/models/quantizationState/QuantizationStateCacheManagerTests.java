/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.SneakyThrows;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN990Codec.KNN990QuantizationStateReader;

import static org.mockito.Mockito.times;

public class QuantizationStateCacheManagerTests extends KNNTestCase {

    @SneakyThrows
    public void testRebuildCache() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            QuantizationStateCache quantizationStateCache = Mockito.mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).rebuildCache();
            QuantizationStateCacheManager.getInstance().rebuildCache();
            Mockito.verify(quantizationStateCache, times(1)).rebuildCache();
        }
    }

    @SneakyThrows
    public void testGetQuantizationState() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            QuantizationStateReadConfig quantizationStateReadConfig = Mockito.mock(QuantizationStateReadConfig.class);
            String cacheKey = "test-key";
            Mockito.when(quantizationStateReadConfig.getCacheKey()).thenReturn(cacheKey);
            QuantizationState quantizationState = Mockito.mock(QuantizationState.class);
            QuantizationStateCache quantizationStateCache = Mockito.mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).addQuantizationState(cacheKey, quantizationState);
            try (MockedStatic<KNN990QuantizationStateReader> mockedStaticReader = Mockito.mockStatic(KNN990QuantizationStateReader.class)) {
                mockedStaticReader.when(() -> KNN990QuantizationStateReader.read(quantizationStateReadConfig))
                    .thenReturn(quantizationState);
                QuantizationStateCacheManager.getInstance().getQuantizationState(quantizationStateReadConfig);
                Mockito.verify(quantizationStateCache, times(1)).addQuantizationState(cacheKey, quantizationState);
            }
            Mockito.when(quantizationStateCache.getQuantizationState(cacheKey)).thenReturn(quantizationState);
            QuantizationStateCacheManager.getInstance().getQuantizationState(quantizationStateReadConfig);
            Mockito.verify(quantizationStateCache, times(1)).addQuantizationState(cacheKey, quantizationState);
        }
    }

    @SneakyThrows
    public void testEvict() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            String field = "test-field";
            QuantizationStateCache quantizationStateCache = Mockito.mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).evict(field);
            QuantizationStateCacheManager.getInstance().evict(field);
            Mockito.verify(quantizationStateCache, times(1)).evict(field);
        }
    }

    @SneakyThrows
    public void testAddQuantizationState() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            String field = "test-field";
            QuantizationState quantizationState = Mockito.mock(QuantizationState.class);
            QuantizationStateCache quantizationStateCache = Mockito.mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).addQuantizationState(field, quantizationState);
            QuantizationStateCacheManager.getInstance().addQuantizationState(field, quantizationState);
            Mockito.verify(quantizationStateCache, times(1)).addQuantizationState(field, quantizationState);
        }
    }

    @SneakyThrows
    public void testSetMaxCacheSizeInKB() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            long maxCacheSizeInKB = 1024;
            QuantizationStateCache quantizationStateCache = Mockito.mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).setMaxCacheSizeInKB(maxCacheSizeInKB);
            QuantizationStateCacheManager.getInstance().setMaxCacheSizeInKB(1024);
            Mockito.verify(quantizationStateCache, times(1)).setMaxCacheSizeInKB(1024);
        }
    }

    @SneakyThrows
    public void testClear() {
        try (MockedStatic<QuantizationStateCache> mockedStaticCache = Mockito.mockStatic(QuantizationStateCache.class)) {
            QuantizationStateCache quantizationStateCache = Mockito.mock(QuantizationStateCache.class);
            mockedStaticCache.when(QuantizationStateCache::getInstance).thenReturn(quantizationStateCache);
            Mockito.doNothing().when(quantizationStateCache).clear();
            QuantizationStateCacheManager.getInstance().clear();
            Mockito.verify(quantizationStateCache, times(1)).clear();
        }
    }
}
