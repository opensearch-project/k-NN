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
/*
 *   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package org.opensearch.knn.plugin.stats.suppliers;
import com.google.common.cache.CacheStats;
import org.opensearch.knn.index.KNNIndexCache;

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
        return getter.apply(KNNIndexCache.getInstance().getStats());
    }
}