/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A cache that contains quantization states
 */
public class QuantizationStateCache {

    private final ConcurrentHashMap<String, QuantizationState> cache = new ConcurrentHashMap<>();
    private final Lock lock = new ReentrantLock();
    private static QuantizationStateCache instance;

    private QuantizationStateCache() {}

    /**
     * Gets the static instance of the cache
     * @return QuantizationStateCache
     */
    public static QuantizationStateCache getInstance() {
        if (instance == null) {
            instance = new QuantizationStateCache();
        }
        return instance;
    }

    /**
     * Gets the quantization state for a given field name
     * @param fieldName field name
     * @return quantization state
     */
    public QuantizationState getQuantizationState(String fieldName) {
        return cache.get(fieldName);
    }

    /**
     * Adds a quantization state to the cache
     * @param fieldName field name
     * @param quantizationState quantization state
     */
    public void addQuantizationState(String fieldName, QuantizationState quantizationState) {
        lock.lock();
        try {
            cache.put(fieldName, quantizationState);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Removes the quantization state associated with a given field name
     * @param fieldName field name
     */
    public void evict(String fieldName) {
        lock.lock();
        try {
            cache.remove(fieldName);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Clears the cache
     */
    public void clear() {
        cache.clear();
    }
}
