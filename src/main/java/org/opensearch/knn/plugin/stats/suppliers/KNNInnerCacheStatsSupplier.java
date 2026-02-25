/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import com.google.common.cache.CacheStats;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Supplier for stats of the cache that the KNNCache uses
 */
public class KNNInnerCacheStatsSupplier implements Supplier<Long> {
    Function<CacheStats, Long> getter;

    /**
     * Constructor
     *
     * @param getter CacheStats method to supply a value
     */
    public KNNInnerCacheStatsSupplier(Function<CacheStats, Long> getter) {
        this.getter = getter;
    }

    @Override
    public Long get() {
        return getter.apply(NativeMemoryCacheManager.getInstance().getCacheStats());
    }
}
