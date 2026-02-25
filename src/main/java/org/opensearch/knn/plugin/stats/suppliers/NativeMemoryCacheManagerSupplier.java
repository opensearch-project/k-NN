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

import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Supplier for stats of KNNIndexCache
 */
public class NativeMemoryCacheManagerSupplier<T> implements Supplier<T> {
    private Function<NativeMemoryCacheManager, T> getter;

    /**
     * Constructor
     *
     * @param getter KNNIndexCache Method to supply a value
     */
    public NativeMemoryCacheManagerSupplier(Function<NativeMemoryCacheManager, T> getter) {
        this.getter = getter;
    }

    @Override
    public T get() {
        return getter.apply(NativeMemoryCacheManager.getInstance());
    }
}
