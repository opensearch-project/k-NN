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

package org.opensearch.knn.plugin.stats;

import com.google.common.cache.CacheStats;
import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.plugin.stats.suppliers.LibraryInitializedSupplier;
import org.opensearch.knn.plugin.stats.suppliers.EventOccurredWithinThresholdSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNCircuitBreakerSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNCounterSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNInnerCacheStatsSupplier;
import org.opensearch.knn.plugin.stats.suppliers.ModelIndexStatusSupplier;
import org.opensearch.knn.plugin.stats.suppliers.ModelIndexingDegradingSupplier;
import org.opensearch.knn.plugin.stats.suppliers.NativeMemoryCacheManagerSupplier;

import java.time.temporal.ChronoUnit;
import java.util.Map;

public class KNNStatsConfig {
    public static  Map<String, KNNStat<?>> KNN_STATS = ImmutableMap.<String, KNNStat<?>>builder()
            .put(StatNames.HIT_COUNT.getName(), new KNNStat<>(false,
                    new KNNInnerCacheStatsSupplier(CacheStats::hitCount)))
            .put(StatNames.MISS_COUNT.getName(), new KNNStat<>(false,
                    new KNNInnerCacheStatsSupplier(CacheStats::missCount)))
            .put(StatNames.LOAD_SUCCESS_COUNT.getName(), new KNNStat<>(false,
                    new KNNInnerCacheStatsSupplier(CacheStats::loadSuccessCount)))
            .put(StatNames.LOAD_EXCEPTION_COUNT.getName(), new KNNStat<>(false,
                    new KNNInnerCacheStatsSupplier(CacheStats::loadExceptionCount)))
            .put(StatNames.TOTAL_LOAD_TIME.getName(), new KNNStat<>(false,
                    new KNNInnerCacheStatsSupplier(CacheStats::totalLoadTime)))
            .put(StatNames.EVICTION_COUNT.getName(), new KNNStat<>(false,
                    new KNNInnerCacheStatsSupplier(CacheStats::evictionCount)))
            .put(StatNames.GRAPH_MEMORY_USAGE.getName(), new KNNStat<>(false,
                    new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getCacheSizeInKilobytes)))
            .put(StatNames.GRAPH_MEMORY_USAGE_PERCENTAGE.getName(), new KNNStat<>(false,
                    new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getCacheSizeAsPercentage)))
            .put(StatNames.INDICES_IN_CACHE.getName(), new KNNStat<>(false,
                    new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getIndicesCacheStats)))
            .put(StatNames.CACHE_CAPACITY_REACHED.getName(), new KNNStat<>(false,
                    new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::isCacheCapacityReached)))
            .put(StatNames.GRAPH_QUERY_ERRORS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.GRAPH_QUERY_ERRORS)))
            .put(StatNames.GRAPH_QUERY_REQUESTS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.GRAPH_QUERY_REQUESTS)))
            .put(StatNames.GRAPH_INDEX_ERRORS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.GRAPH_INDEX_ERRORS)))
            .put(StatNames.GRAPH_INDEX_REQUESTS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.GRAPH_INDEX_REQUESTS)))
            .put(StatNames.CIRCUIT_BREAKER_TRIGGERED.getName(), new KNNStat<>(true,
                    new KNNCircuitBreakerSupplier()))
            .put(StatNames.MODEL_INDEX_STATUS.getName(), new KNNStat<>(true,
                    new ModelIndexStatusSupplier<>(ModelDao::getHealthStatus)))
            .put(StatNames.KNN_QUERY_REQUESTS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.KNN_QUERY_REQUESTS)))
            .put(StatNames.SCRIPT_COMPILATIONS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.SCRIPT_COMPILATIONS)))
            .put(StatNames.SCRIPT_COMPILATION_ERRORS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.SCRIPT_COMPILATION_ERRORS)))
            .put(StatNames.SCRIPT_QUERY_REQUESTS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.SCRIPT_QUERY_REQUESTS)))
            .put(StatNames.SCRIPT_QUERY_ERRORS.getName(), new KNNStat<>(false,
                    new KNNCounterSupplier(KNNCounter.SCRIPT_QUERY_ERRORS)))
            .put(StatNames.INDEXING_FROM_MODEL_DEGRADED.getName(), new KNNStat<>(false,
                    new EventOccurredWithinThresholdSupplier(
                        new ModelIndexingDegradingSupplier(ModelCache::getEvictedDueToSizeAt),
                        KNNConstants.MODEL_CACHE_CAPACITY_ATROPHY_THRESHOLD_IN_MINUTES,
                        ChronoUnit.MINUTES)))
            .put(StatNames.FAISS_LOADED.getName(), new KNNStat<>(false,
                    new LibraryInitializedSupplier(KNNEngine.FAISS)))
            .put(StatNames.NMSLIB_LOADED.getName(), new KNNStat<>(false,
                    new LibraryInitializedSupplier(KNNEngine.NMSLIB)))
            .build();
}
