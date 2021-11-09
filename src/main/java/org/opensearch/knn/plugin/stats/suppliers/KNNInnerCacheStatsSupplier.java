/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
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