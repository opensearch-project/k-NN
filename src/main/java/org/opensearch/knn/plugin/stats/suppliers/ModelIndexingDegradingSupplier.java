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

import org.opensearch.knn.indices.ModelCache;

import java.time.Instant;
import java.util.function.Function;
import java.util.function.Supplier;

public class ModelIndexingDegradingSupplier implements Supplier<Instant> {

    private final Function<ModelCache, Instant> getter;

    public ModelIndexingDegradingSupplier(Function<ModelCache, Instant> getter) {
        this.getter = getter;
    }

    @Override
    public Instant get() {
        return getter.apply(ModelCache.getInstance());
    }
}
