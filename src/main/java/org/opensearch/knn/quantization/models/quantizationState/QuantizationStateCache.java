/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;

/**
 * A thread-safe singleton cache that contains quantization states.
 */
public class QuantizationStateCache {

    private static volatile QuantizationStateCache instance;
    private final Cache<String, QuantizationState> cache;

    private QuantizationStateCache() {
        this.cache = CacheBuilder.newBuilder().build();
    }

    /**
     * Gets the singleton instance of the cache.
     * @return QuantizationStateCache
     */
    public static QuantizationStateCache getInstance() {
        if (instance == null) {
            synchronized (QuantizationStateCache.class) {
                if (instance == null) {
                    instance = new QuantizationStateCache();
                }
            }
        }
        return instance;
    }

    /**
     * Retrieves the quantization state associated with a given field name.
     * @param fieldName The name of the field.
     * @return The associated QuantizationState, or null if not present.
     */
    public QuantizationState getQuantizationState(String fieldName) {
        return cache.getIfPresent(fieldName);
    }

    /**
     * Adds or updates a quantization state in the cache.
     * @param fieldName The name of the field.
     * @param quantizationState The quantization state to store.
     */
    public void addQuantizationState(String fieldName, QuantizationState quantizationState) {
        cache.put(fieldName, quantizationState);
    }

    /**
     * Removes the quantization state associated with a given field name.
     * @param fieldName The name of the field.
     */
    public void evict(String fieldName) {
        cache.invalidate(fieldName);
    }

    /**
     * Clears all entries from the cache.
     */
    public void clear() {
        cache.invalidateAll();
    }
}
