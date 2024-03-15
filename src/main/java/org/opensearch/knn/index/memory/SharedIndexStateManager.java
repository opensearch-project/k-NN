/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import com.google.common.annotations.VisibleForTesting;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.jni.JNIService;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Class manages allocations that can be shared between native indices. No locking is required.
 * Once a caller obtain an instance of a {@link org.opensearch.knn.index.memory.SharedIndexState}, it is guaranteed to
 * be valid until it is returned. {@link org.opensearch.knn.index.memory.SharedIndexState} are reference counted
 * internally. Once the reference count goes to 0, it will be freed.
 */
@Log4j2
class SharedIndexStateManager {
    // Map storing the shared index state with key being the modelId.
    private final ConcurrentHashMap<String, SharedIndexStateEntry> sharedIndexStateCache;
    private final ReadWriteLock readWriteLock;

    private static SharedIndexStateManager INSTANCE;

    // TODO: Going to refactor away from doing this in the future. For now, keeping for simplicity.
    public static synchronized SharedIndexStateManager getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new SharedIndexStateManager();
        }
        return INSTANCE;
    }

    /**
     * Constructor
     */
    @VisibleForTesting
    SharedIndexStateManager() {
        this.sharedIndexStateCache = new ConcurrentHashMap<>();
        this.readWriteLock = new ReentrantReadWriteLock();
    }

    /**
     * Return a {@link SharedIndexState} associated with the key. If no value exists, it will attempt to create it.
     * Once returned, the {@link SharedIndexState} will be valid until
     * {@link SharedIndexStateManager#release(SharedIndexState)} is called. Caller must ensure that this is
     * called after it is done using it.
     *
     * In order to create the shared state, it will use the indexAddress passed in to create the shared state from
     * using {@link org.opensearch.knn.jni.JNIService#initSharedIndexState(long, KNNEngine)}.
     *
     * @param indexAddress Address of index to initialize the shared state from
     * @param knnEngine engine index belongs to
     * @return ShareModelContext
     */
    public SharedIndexState get(long indexAddress, String modelId, KNNEngine knnEngine) {
        this.readWriteLock.readLock().lock();
        try {
            // This can be done safely with readLock because the ConcurrentHasMap.computeIfAbsent guarantees:
            //
            // "If the specified key is not already associated with a value, attempts to compute its value using the given
            // mapping function and enters it into this map unless null. The entire method invocation is performed
            // atomically, so the function is applied at most once per key. Some attempted update operations on this map
            // by other threads may be blocked while computation is in progress, so the computation should be short and
            // simple, and must not attempt to update any other mappings of this map."
            //
            // Ref:
            // https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/ConcurrentHashMap.html#computeIfAbsent-K-java.util.function.Function-
            SharedIndexStateEntry entry = sharedIndexStateCache.computeIfAbsent(modelId, m -> {
                log.info("Loading entry to shared index state cache for model {}", modelId);
                long sharedIndexStateAddress = JNIService.initSharedIndexState(indexAddress, knnEngine);
                return new SharedIndexStateEntry(new SharedIndexState(sharedIndexStateAddress, modelId, knnEngine));
            });
            entry.incRef();
            return entry.getSharedIndexState();
        } finally {
            this.readWriteLock.readLock().unlock();
        }
    }

    /**
     * Indicate that the {@link SharedIndexState} is no longer being used. If nothing else is using it, it will be
     * removed from the cache and evicted.
     *
     * After calling this method, {@link SharedIndexState} should no longer be used by calling thread.
     *
     * @param sharedIndexState to return to the system.
     */
    public void release(SharedIndexState sharedIndexState) {
        this.readWriteLock.writeLock().lock();

        try {
            if (!sharedIndexStateCache.containsKey(sharedIndexState.getModelId())) {
                // This should not happen. Will log the error and return to prevent crash
                log.error("Attempting to evict model from cache but it is not present: {}", sharedIndexState.getModelId());
                this.readWriteLock.writeLock().unlock();
                return;
            }

            long refCount = sharedIndexStateCache.get(sharedIndexState.getModelId()).decRef();
            if (refCount <= 0) {
                log.info("Evicting entry from shared index state cache for key {}", sharedIndexState.getModelId());
                sharedIndexStateCache.remove(sharedIndexState.getModelId());
                JNIService.freeSharedIndexState(sharedIndexState.getSharedIndexStateAddress(), sharedIndexState.getKnnEngine());
            }
        } finally {
            this.readWriteLock.writeLock().unlock();
        }
    }

    private static final class SharedIndexStateEntry {
        @Getter
        private final SharedIndexState sharedIndexState;
        private final AtomicLong referenceCount;

        /**
         * Constructor
         *
         * @param sharedIndexState sharedIndexStateContext being wrapped
         */
        private SharedIndexStateEntry(SharedIndexState sharedIndexState) {
            this.sharedIndexState = sharedIndexState;
            this.referenceCount = new AtomicLong(0);
        }

        /**
         * Increases reference count by 1
         *
         * @return ++referenceCount
         */
        private long incRef() {
            return referenceCount.incrementAndGet();
        }

        /**
         * Decrease reference count by 1
         *
         * @return --referenceCount
         */
        private long decRef() {
            return referenceCount.decrementAndGet();
        }
    }
}
