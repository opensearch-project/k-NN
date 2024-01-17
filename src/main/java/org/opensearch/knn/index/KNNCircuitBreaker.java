/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.threadpool.ThreadPool;

import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * Runs the circuit breaker logic and updates the settings
 */
public class KNNCircuitBreaker {
    private static Logger logger = LogManager.getLogger(KNNCircuitBreaker.class);
    public static int CB_TIME_INTERVAL = 2 * 60; // seconds

    private static KNNCircuitBreaker INSTANCE;
    private ThreadPool threadPool;
    private ClusterService clusterService;

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

    public void initialize(ThreadPool threadPool, ClusterService clusterService) {
        this.threadPool = threadPool;
        this.clusterService = clusterService;
        NativeMemoryCacheManager nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        Runnable runnable = () -> {
            if (nativeMemoryCacheManager.isCacheCapacityReached() && clusterService.localNode().isDataNode()) {
                long currentSizeKiloBytes = nativeMemoryCacheManager.getCacheSizeInKilobytes();
                long circuitBreakerLimitSizeKiloBytes = KNNSettings.getCircuitBreakerLimit().getKb();
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
                List<String> nodesAtMaxCapacity;
                try {
                    nodesAtMaxCapacity = KNNCircuitBreakerUtil.instance().getNodesAtMaxCapacity();
                } catch (ExecutionException | InterruptedException e) {
                    logger.error("Unable to get knn stats and determine if any nodes are at capacity", e);
                    return;
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
                    KNNCircuitBreakerUtil.instance().updateCircuitBreakerSettings(false);
                }
            }
        };
        this.threadPool.scheduleWithFixedDelay(runnable, TimeValue.timeValueSeconds(CB_TIME_INTERVAL), ThreadPool.Names.GENERIC);
    }
}
