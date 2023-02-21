/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory.breaker;

import lombok.Value;
import org.opensearch.common.component.AbstractLifecycleComponent;
import org.opensearch.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;
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
import org.opensearch.threadpool.Scheduler;
import org.opensearch.threadpool.ThreadPool;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Service handling native memory circuit breaking logic. The circuit breaker gets tripped based on memory demand
 * tracked by the {@link NativeMemoryCacheManager}. When {@link NativeMemoryCacheManager}'s cache fills up, if the
 * circuit breaking logic is enabled, it will trip the circuit. Elsewhere in the code, the circuit breaker's value can
 * be queried to prevent actions that should not happen during high memory pressure.
 */
public class NativeMemoryCircuitBreakerService extends AbstractLifecycleComponent {
    private final ThreadPool threadPool;
    private final KNNSettings knnSettings;
    private static final Logger logger = LogManager.getLogger(NativeMemoryCircuitBreakerService.class);
    public static int CB_TIME_INTERVAL = 2 * 60; // seconds
    private final ClusterService clusterService;
    private final Client client;
    // Cancellable task to track circuitBreakerRunnable. In order to schedule, doStart must be called. doStart will
    // only start the future if the previous value is null. Therefore, to close this class, do NOT set the value of
    // this variable to null. To stop this class, this variable should be set to null so that it may be restarted.
    private final AtomicReference<Scheduler.Cancellable> circuitBreakerFuture;

    /**
     * Constructor for creation of circuit breaker service for KNN
     *
     * @param knnSettings Settings class for k-NN
     * @param threadPool thread pool for circuit breaker monitor to run job
     * @param clusterService cluster service used to retrieve information about the cluster
     * @param client client used to make calls to the cluster
     */
    public NativeMemoryCircuitBreakerService(KNNSettings knnSettings, ThreadPool threadPool, ClusterService clusterService, Client client) {
        this.knnSettings = knnSettings;
        this.threadPool = threadPool;
        this.clusterService = clusterService;
        this.client = client;
        this.circuitBreakerFuture = new AtomicReference<>(null);
    }

    /**
     * Checks if the circuit breaker is triggered
     *
     * @return true if circuit breaker is triggered; false otherwise
     */
    public boolean isCircuitBreakerTriggered() {
        return knnSettings.getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED);
    }

    /**
     * Sets circuit breaker to new value
     *
     * @param circuitBreaker value to update circuit breaker to
     */
    public void setCircuitBreaker(boolean circuitBreaker) {
        knnSettings.updateBooleanSetting(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED, circuitBreaker);
    }

    /**
     * Gets the limit of the circuit breaker
     *
     * @return limit as ByteSizeValue of native memory circuit breaker
     */
    public ByteSizeValue getCircuitBreakerLimit() {
        return knnSettings.getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT);
    }

    /**
     * Determine if the circuit breaker is enabled
     *
     * @return true if circuit breaker is enabled. False otherwise.
     */
    public boolean isCircuitBreakerEnabled() {
        return knnSettings.getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED);
    }

    /**
     * Returns the percentage as a double for when to unset the circuit breaker
     *
     * @return percentage as double for unsetting circuit breaker
     */
    private double getCircuitBreakerUnsetPercentage() {
        return knnSettings.getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE);
    }

    @Override
    protected void doStart() {
        Monitor monitor = new Monitor(this, NativeMemoryCacheManager.getInstance(), clusterService, client);
        this.circuitBreakerFuture.compareAndSet(
            null,
            threadPool.scheduleWithFixedDelay(monitor, TimeValue.timeValueSeconds(CB_TIME_INTERVAL), ThreadPool.Names.GENERIC)
        );
    }

    @Override
    protected void doStop() {
        Scheduler.Cancellable cancellable = this.circuitBreakerFuture.getAndSet(null);
        if (cancellable != null && !cancellable.isCancelled()) {
            cancellable.cancel();
        }
    }

    @Override
    protected void doClose() {
        Scheduler.Cancellable cancellable = this.circuitBreakerFuture.get();
        if (cancellable != null && !cancellable.isCancelled()) {
            cancellable.cancel();
        }
    }

    @Value
    private static class Monitor implements Runnable {
        NativeMemoryCircuitBreakerService nativeMemoryCircuitBreakerService;
        NativeMemoryCacheManager nativeMemoryCacheManager;
        ClusterService clusterService;
        Client client;

        @Override
        public void run() {
            if (nativeMemoryCacheManager.isCacheCapacityReached() && clusterService.localNode().isDataNode()) {
                long currentSizeKiloBytes = nativeMemoryCacheManager.getCacheSizeInKilobytes();
                long circuitBreakerLimitSizeKiloBytes = nativeMemoryCircuitBreakerService.getCircuitBreakerLimit().getKb();
                long circuitBreakerUnsetSizeKiloBytes = (long) ((nativeMemoryCircuitBreakerService.getCircuitBreakerUnsetPercentage() / 100)
                    * circuitBreakerLimitSizeKiloBytes);
                // Unset capacityReached flag if currentSizeBytes is less than circuitBreakerUnsetSizeBytes
                if (currentSizeKiloBytes <= circuitBreakerUnsetSizeKiloBytes) {
                    nativeMemoryCacheManager.setCacheCapacityReached(false);
                }
            }

            // Leader node untriggers CB if all nodes have not reached their max capacity
            if (nativeMemoryCircuitBreakerService.isCircuitBreakerTriggered()
                && clusterService.state().nodes().isLocalNodeElectedClusterManager()) {
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
                        nativeMemoryCircuitBreakerService.setCircuitBreaker(false);
                    }
                } catch (Exception e) {
                    logger.error("[KNN] Exception getting stats: " + e);
                }
            }
        }
    }
}
