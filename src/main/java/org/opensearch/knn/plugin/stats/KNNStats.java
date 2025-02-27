/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import com.google.common.cache.CacheStats;
import com.google.common.collect.ImmutableMap;
import org.opensearch.Version;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.plugin.stats.suppliers.EventOccurredWithinThresholdSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNNodeLevelCircuitBreakerSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNCounterSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNInnerCacheStatsSupplier;
import org.opensearch.knn.plugin.stats.suppliers.LibraryInitializedSupplier;
import org.opensearch.knn.plugin.stats.suppliers.ModelIndexStatusSupplier;
import org.opensearch.knn.plugin.stats.suppliers.ModelIndexingDegradingSupplier;
import org.opensearch.knn.plugin.stats.suppliers.NativeMemoryCacheManagerSupplier;
import org.opensearch.transport.client.Client;

import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Class represents all stats the plugin keeps track of
 */
public class KNNStats {

    private final Client client;
    private final Supplier<Version> minVersionSupplier;
    private final Map<String, KNNStat<?>> knnStats;

    /**
     * Constructor
     */
    public KNNStats(Client client, Supplier<Version> minVersionSupplier) {
        this.client = client;
        this.minVersionSupplier = minVersionSupplier;
        this.knnStats = buildStatsMap();
    }

    /**
     * Get the stats
     *
     * @return all of the stats
     */
    public Map<String, KNNStat<?>> getStats() {
        return knnStats;
    }

    /**
     * Get a map of the stats that are kept at the node level
     *
     * @return Map of stats kept at the node level
     */
    public Map<String, KNNStat<?>> getNodeStats() {
        return getClusterOrNodeStats(false);
    }

    /**
     * Get a map of the stats that are kept at the cluster level
     *
     * @return Map of stats kept at the cluster level
     */
    public Map<String, KNNStat<?>> getClusterStats() {
        return getClusterOrNodeStats(true);
    }

    private Map<String, KNNStat<?>> getClusterOrNodeStats(Boolean getClusterStats) {
        Map<String, KNNStat<?>> statsMap = new HashMap<>();

        for (Map.Entry<String, KNNStat<?>> entry : knnStats.entrySet()) {
            if (entry.getValue().isClusterLevel() == getClusterStats) {
                statsMap.put(entry.getKey(), entry.getValue());
            }
        }
        return statsMap;
    }

    private Map<String, KNNStat<?>> buildStatsMap() {
        ImmutableMap.Builder<String, KNNStat<?>> builder = ImmutableMap.<String, KNNStat<?>>builder();
        addQueryStats(builder);
        addNativeMemoryStats(builder);
        addEngineStats(builder);
        addScriptStats(builder);
        addModelStats(builder);
        addGraphStats(builder);
        return builder.build();
    }

    private void addQueryStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        // KNN Query Stats
        builder.put(StatNames.KNN_QUERY_REQUESTS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_REQUESTS)))
            .put(
                StatNames.KNN_QUERY_WITH_FILTER_REQUESTS.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.KNN_QUERY_WITH_FILTER_REQUESTS))
            );

        // Min Score Query Stats
        builder.put(
            StatNames.MIN_SCORE_QUERY_REQUESTS.getName(),
            new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.MIN_SCORE_QUERY_REQUESTS))
        )
            .put(
                StatNames.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS))
            );

        // Max Distance Query Stats
        builder.put(
            StatNames.MAX_DISTANCE_QUERY_REQUESTS.getName(),
            new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.MAX_DISTANCE_QUERY_REQUESTS))
        )
            .put(
                StatNames.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS))
            );
    }

    private void addNativeMemoryStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.HIT_COUNT.getName(), new KNNStat<>(false, new KNNInnerCacheStatsSupplier(CacheStats::hitCount)))
            .put(StatNames.MISS_COUNT.getName(), new KNNStat<>(false, new KNNInnerCacheStatsSupplier(CacheStats::missCount)))
            .put(StatNames.LOAD_SUCCESS_COUNT.getName(), new KNNStat<>(false, new KNNInnerCacheStatsSupplier(CacheStats::loadSuccessCount)))
            .put(
                StatNames.LOAD_EXCEPTION_COUNT.getName(),
                new KNNStat<>(false, new KNNInnerCacheStatsSupplier(CacheStats::loadExceptionCount))
            )
            .put(StatNames.TOTAL_LOAD_TIME.getName(), new KNNStat<>(false, new KNNInnerCacheStatsSupplier(CacheStats::totalLoadTime)))
            .put(StatNames.EVICTION_COUNT.getName(), new KNNStat<>(false, new KNNInnerCacheStatsSupplier(CacheStats::evictionCount)))
            .put(
                StatNames.GRAPH_MEMORY_USAGE.getName(),
                new KNNStat<>(false, new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getIndicesSizeInKilobytes))
            )
            .put(
                StatNames.GRAPH_MEMORY_USAGE_PERCENTAGE.getName(),
                new KNNStat<>(false, new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getIndicesSizeAsPercentage))
            )
            .put(
                StatNames.INDICES_IN_CACHE.getName(),
                new KNNStat<>(false, new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getIndicesCacheStats))
            )
            .put(StatNames.CACHE_CAPACITY_REACHED.getName(), new KNNStat<>(false, new KNNNodeLevelCircuitBreakerSupplier()))
            .put(StatNames.GRAPH_QUERY_ERRORS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.GRAPH_QUERY_ERRORS)))
            .put(StatNames.GRAPH_QUERY_REQUESTS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.GRAPH_QUERY_REQUESTS)))
            .put(StatNames.GRAPH_INDEX_ERRORS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.GRAPH_INDEX_ERRORS)))
            .put(StatNames.GRAPH_INDEX_REQUESTS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.GRAPH_INDEX_REQUESTS)))
            .put(StatNames.CIRCUIT_BREAKER_TRIGGERED.getName(), new CircuitBreakerStat(client, minVersionSupplier));
    }

    private void addEngineStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.FAISS_LOADED.getName(), new KNNStat<>(false, new LibraryInitializedSupplier(KNNEngine.FAISS)))
            .put(StatNames.NMSLIB_LOADED.getName(), new KNNStat<>(false, new LibraryInitializedSupplier(KNNEngine.NMSLIB)))
            .put(StatNames.LUCENE_LOADED.getName(), new KNNStat<>(false, new LibraryInitializedSupplier(KNNEngine.LUCENE)));
    }

    private void addScriptStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.SCRIPT_COMPILATIONS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.SCRIPT_COMPILATIONS)))
            .put(
                StatNames.SCRIPT_COMPILATION_ERRORS.getName(),
                new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.SCRIPT_COMPILATION_ERRORS))
            )
            .put(StatNames.SCRIPT_QUERY_REQUESTS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.SCRIPT_QUERY_REQUESTS)))
            .put(StatNames.SCRIPT_QUERY_ERRORS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.SCRIPT_QUERY_ERRORS)));
    }

    private void addModelStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(
            StatNames.INDEXING_FROM_MODEL_DEGRADED.getName(),
            new KNNStat<>(
                false,
                new EventOccurredWithinThresholdSupplier(
                    new ModelIndexingDegradingSupplier(ModelCache::getEvictedDueToSizeAt),
                    KNNConstants.MODEL_CACHE_CAPACITY_ATROPHY_THRESHOLD_IN_MINUTES,
                    ChronoUnit.MINUTES
                )
            )
        )
            .put(StatNames.MODEL_INDEX_STATUS.getName(), new KNNStat<>(true, new ModelIndexStatusSupplier<>(ModelDao::getHealthStatus)))
            .put(StatNames.TRAINING_REQUESTS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.TRAINING_REQUESTS)))
            .put(StatNames.TRAINING_ERRORS.getName(), new KNNStat<>(false, new KNNCounterSupplier(KNNCounter.TRAINING_ERRORS)))
            .put(
                StatNames.TRAINING_MEMORY_USAGE.getName(),
                new KNNStat<>(false, new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getTrainingSizeInKilobytes))
            )
            .put(
                StatNames.TRAINING_MEMORY_USAGE_PERCENTAGE.getName(),
                new KNNStat<>(false, new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getTrainingSizeAsPercentage))
            );
    }

    private void addGraphStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.GRAPH_STATS.getName(), new KNNStat<>(false, new Supplier<Map<String, Map<String, Object>>>() {
            @Override
            public Map<String, Map<String, Object>> get() {
                return createGraphStatsMap();
            }
        }));
    }

    private Map<String, Map<String, Object>> createGraphStatsMap() {
        Map<String, Object> mergeMap = new HashMap<>();
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_OPERATIONS.getName(), KNNGraphValue.MERGE_CURRENT_OPERATIONS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_DOCS.getName(), KNNGraphValue.MERGE_CURRENT_DOCS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.getName(), KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_OPERATIONS.getName(), KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getName(), KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_DOCS.getName(), KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getName(), KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());
        Map<String, Object> refreshMap = new HashMap<>();
        refreshMap.put(KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getName(), KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        refreshMap.put(KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getName(), KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue());
        Map<String, Map<String, Object>> graphStatsMap = new HashMap<>();
        graphStatsMap.put(StatNames.MERGE.getName(), mergeMap);
        graphStatsMap.put(StatNames.REFRESH.getName(), refreshMap);
        return graphStatsMap;
    }
}
