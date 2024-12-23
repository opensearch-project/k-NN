/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.RemovalCause;
import com.google.common.cache.RemovalNotification;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.util.ScheduledExecutor;

import java.io.Closeable;
import java.io.IOException;
import java.time.Instant;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.index.KNNSettings.QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES;
import static org.opensearch.knn.index.KNNSettings.QUANTIZATION_STATE_CACHE_SIZE_LIMIT;

/**
 * A thread-safe singleton cache that contains quantization states.
 */
@Log4j2
public class QuantizationStateCache implements Closeable {

    private static volatile QuantizationStateCache instance;
    private Cache<String, QuantizationState> cache;
    private ScheduledExecutor cacheMaintainer;
    @Getter
    private long maxCacheSizeInKB;
    @Getter
    private Instant evictedDueToSizeAt;

    @VisibleForTesting
    QuantizationStateCache() {
        maxCacheSizeInKB = ((ByteSizeValue) KNNSettings.state().getSettingValue(QUANTIZATION_STATE_CACHE_SIZE_LIMIT)).getKb();
        buildCache();
    }

    /**
     * Gets the singleton instance of the cache.
     * @return QuantizationStateCache
     */
    static QuantizationStateCache getInstance() {
        if (instance == null) {
            synchronized (QuantizationStateCache.class) {
                if (instance == null) {
                    instance = new QuantizationStateCache();
                }
            }
        }
        return instance;
    }

    private void buildCache() {
        if (cacheMaintainer != null) {
            cacheMaintainer.close();
        }

        this.cache = CacheBuilder.newBuilder().concurrencyLevel(1).maximumWeight(maxCacheSizeInKB).weigher((k, v) -> {
            try {
                return ((QuantizationState) v).toByteArray().length;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        })
            .expireAfterAccess(
                ((TimeValue) KNNSettings.state().getSettingValue(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES)).getMinutes(),
                TimeUnit.MINUTES
            )
            .removalListener(this::onRemoval)
            .build();

        Runnable cleanUp = () -> {
            try {
                cache.cleanUp();
            } catch (Exception e) {
                // Exceptions from Guava shouldn't halt the executor
                log.error("Error cleaning up cache", e);
            }
        };
        long scheduleMillis = ((TimeValue) KNNSettings.state().getSettingValue(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES)).getMillis();
        this.cacheMaintainer = new ScheduledExecutor(Executors.newSingleThreadScheduledExecutor(), cleanUp, scheduleMillis);
    }

    synchronized void rebuildCache() {
        clear();
        buildCache();
    }

    /**
     * Retrieves the quantization state associated with a given field name.
     * @param fieldName The name of the field.
     * @return The associated QuantizationState, or null if not present.
     */
    QuantizationState getQuantizationState(String fieldName) {
        return cache.getIfPresent(fieldName);
    }

    /**
     * Adds or updates a quantization state in the cache.
     * @param fieldName The name of the field.
     * @param quantizationState The quantization state to store.
     */
    void addQuantizationState(String fieldName, QuantizationState quantizationState) {
        cache.put(fieldName, quantizationState);
    }

    /**
     * Removes the quantization state associated with a given field name.
     * @param fieldName The name of the field.
     */
    public void evict(String fieldName) {
        cache.invalidate(fieldName);
    }

    private void onRemoval(RemovalNotification<String, QuantizationState> removalNotification) {
        if (RemovalCause.SIZE == removalNotification.getCause()) {
            updateEvictedDueToSizeAt();
            log.info(
                "[KNN] Quantization state evicted from cache. Key {}, Reason: {}",
                removalNotification.getKey(),
                removalNotification.getCause()
            );
        }
    }

    void setMaxCacheSizeInKB(long maxCacheSizeInKB) {
        this.maxCacheSizeInKB = maxCacheSizeInKB;
    }

    private void updateEvictedDueToSizeAt() {
        evictedDueToSizeAt = Instant.now();
    }

    /**
     * Clears all entries from the cache.
     */
    public void clear() {
        cache.invalidateAll();
    }

    @Override
    public void close() throws IOException {
        if (cacheMaintainer != null) {
            cacheMaintainer.close();
        }
    }
}
