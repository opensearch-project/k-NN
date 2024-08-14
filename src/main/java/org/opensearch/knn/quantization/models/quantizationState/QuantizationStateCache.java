/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class QuantizationStateCache {

    private final ConcurrentHashMap<String, QuantizationState> cache = new ConcurrentHashMap<>();
    private final Lock lock = new ReentrantLock();
    private static QuantizationStateCache instance;

    private QuantizationStateCache() {}

    public static QuantizationStateCache getInstance() {
        if (instance == null) {
            instance = new QuantizationStateCache();
        }
        return instance;
    }

    public QuantizationState getQuantizationState(String fieldName) {
        return cache.get(fieldName);
    }

    public void addQuantizationState(String fieldName, QuantizationState quantizationState) {
        lock.lock();
        try {
            cache.put(fieldName, quantizationState);
        } finally {
            lock.unlock();
        }
    }

    public void evict(String fieldName) {
        lock.lock();
        try {
            cache.remove(fieldName);
        } finally {
            lock.unlock();
        }
    }

    public void clear() {
        cache.clear();
    }
}
