/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.cache.Cache;

import java.io.Closeable;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Performs periodic maintenance for a Guava cache. The Guava cache is implemented in a way that maintenance operations (such as evicting expired
 * entries) will only occur when the cache is accessed. See {@see <a href="https://github.com/google/guava/wiki/cachesexplained#timed-eviction"> Guava Cache Guide</a>}
 * for more details. Thus, to perform any pending maintenance, the cleanUp method will be called periodically from a CacheMaintainer instance.
 */
public class CacheMaintainer<K, V> implements Closeable {
    private final Cache<K, V> cache;
    private final ScheduledExecutorService executor;
    private static final int DEFAULT_INTERVAL_SECONDS = 60;

    public CacheMaintainer(Cache<K, V> cache) {
        this.cache = cache;
        this.executor = Executors.newSingleThreadScheduledExecutor();
    }

    public void startMaintenance() {
        executor.scheduleAtFixedRate(this::cleanCache, DEFAULT_INTERVAL_SECONDS, DEFAULT_INTERVAL_SECONDS, TimeUnit.SECONDS);
    }

    public void cleanCache() {
        cache.cleanUp();
    }

    @Override
    public void close() {
        executor.shutdown();
    }
}
