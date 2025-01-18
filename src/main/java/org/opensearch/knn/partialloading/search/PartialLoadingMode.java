/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import java.util.function.Supplier;

public enum PartialLoadingMode {
    DISABLED(null),
    MEMORY_EFFICIENT(MemoryEfficientPartialLoadingSearchStrategy::new);

    PartialLoadingMode(Supplier<PartialLoadingSearchStrategy> searchStrategyFactory) {
        this.searchStrategyFactory = searchStrategyFactory;
    }

    public PartialLoadingSearchStrategy createSearchStrategy() {
        return searchStrategyFactory.get();
    }

    private final Supplier<PartialLoadingSearchStrategy> searchStrategyFactory;
}
