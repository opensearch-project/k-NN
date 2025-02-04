/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.common.lifecycle.LifecycleListener;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsNodeResponse;
import org.opensearch.knn.plugin.transport.KNNStatsRequest;
import org.opensearch.knn.plugin.transport.KNNStatsResponse;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.threadpool.ThreadPool;

import java.util.ArrayList;
import java.util.List;

/**
 * Runs the circuit breaker logic and updates the settings
 */
public class KNNCircuitBreaker {
    private static Logger logger = LogManager.getLogger(KNNCircuitBreaker.class);
    public static int CB_TIME_INTERVAL = 10; // seconds

    private static KNNCircuitBreaker INSTANCE;
    private ThreadPool threadPool;
    private ClusterService clusterService;
    private Client client;

    private KNNCircuitBreaker() {}

    public static synchronized KNNCircuitBreaker getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new KNNCircuitBreaker();
        }
        return INSTANCE;
    }

    /**
     * SetInstance of Circuit Breaker
     *
     * @param instance KNNCircuitBreaker instance
     */
    public static synchronized void setInstance(KNNCircuitBreaker instance) {
        INSTANCE = instance;
    }

    public void initialize(ThreadPool threadPool, ClusterService clusterService, Client client) {
        this.threadPool = threadPool;
        this.clusterService = clusterService;
        this.client = client;
        NativeMemoryCacheManager nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();

        Runnable runnable = () -> {
            if (nativeMemoryCacheManager.isCacheCapacityReached() && clusterService.localNode().isDataNode()) {
                long currentSizeKiloBytes = nativeMemoryCacheManager.getCacheSizeInKilobytes();
                long circuitBreakerLimitSizeKiloBytes = KNNSettings.state().getCircuitBreakerLimit().getKb();
                long circuitBreakerUnsetSizeKiloBytes = (long) ((KNNSettings.getCircuitBreakerUnsetPercentage() / 100)
                    * circuitBreakerLimitSizeKiloBytes);
                /**
                 * Unset capacityReached flag if currentSizeBytes is less than circuitBreakerUnsetSizeBytes
                 */
                if (currentSizeKiloBytes <= circuitBreakerUnsetSizeKiloBytes) {
                    nativeMemoryCacheManager.setCacheCapacityReached(false);
                }
            }

            // Leader node untriggers CB if all nodes have not reached their max capacity
            if (KNNSettings.isCircuitBreakerTriggered() && clusterService.state().nodes().isLocalNodeElectedClusterManager()) {
                KNNStatsRequest knnStatsRequest = new KNNStatsRequest();
                knnStatsRequest.addStat(StatNames.CACHE_CAPACITY_REACHED.getName());
                knnStatsRequest.timeout(new TimeValue(1000 * 10)); // 10 second timeout

                try {
                    KNNStatsResponse knnStatsResponse = client.execute(KNNStatsAction.INSTANCE, knnStatsRequest).get();
                    List<KNNStatsNodeResponse> nodeResponses = knnStatsResponse.getNodes();

                    List<String> nodesAtMaxCapacity = new ArrayList<>();
                    for (KNNStatsNodeResponse nodeResponse : nodeResponses) {
                        if ((Boolean) nodeResponse.getStatsMap().get(StatNames.CACHE_CAPACITY_REACHED.getName())) {
                            nodesAtMaxCapacity.add(nodeResponse.getNode().getId());
                        }
                    }

                    if (!nodesAtMaxCapacity.isEmpty()) {
                        logger.info(
                            "[KNN] knn.circuit_breaker.triggered stays set. Nodes at max cache capacity: "
                                + String.join(",", nodesAtMaxCapacity)
                                + "."
                        );
                    } else {
                        logger.info(
                            "[KNN] Cache capacity below 75% of the circuit breaker limit for all nodes."
                                + " Unsetting knn.circuit_breaker.triggered flag."
                        );
                        KNNSettings.state().updateCircuitBreakerSettings(false);
                    }
                } catch (Exception e) {
                    logger.error("[KNN] Exception getting stats: " + e);
                }
            }
        };
        this.threadPool.scheduleWithFixedDelay(runnable, TimeValue.timeValueSeconds(CB_TIME_INTERVAL), ThreadPool.Names.GENERIC);

        // Update when node is fully joined
        clusterService.addLifecycleListener(new LifecycleListener() {
            @Override
            public void afterStart() {
                // Node is fully initialized, rebuild the cache to fetch node-specific limits
                nativeMemoryCacheManager.rebuildCache();
            }
        });
    }
}
