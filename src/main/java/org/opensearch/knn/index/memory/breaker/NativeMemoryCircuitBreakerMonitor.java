/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory.breaker;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsNodeResponse;
import org.opensearch.knn.plugin.transport.KNNStatsRequest;
import org.opensearch.knn.plugin.transport.KNNStatsResponse;
import org.opensearch.threadpool.Scheduler;
import org.opensearch.threadpool.ThreadPool;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Job that runs periodically to monitor native memory usage on the node/in the cluster and untrip the
 * NativeMemoryCircuitBreaker if necessary.
 */
@Log4j2
public class NativeMemoryCircuitBreakerMonitor implements Closeable {
    private final NativeMemoryCircuitBreaker nativeMemoryCircuitBreaker;
    private final NativeMemoryCacheManager nativeMemoryCacheManager;
    private final ClusterService clusterService;
    private final Client client;
    private final ThreadPool threadPool;
    private final AtomicReference<Boolean> isStarted;
    private Scheduler.Cancellable monitorFuture;
    public static final int CB_TIME_INTERVAL = 2 * 60; // seconds
    private static final TimeValue STATS_REQUEST_TIMEOUT = new TimeValue(1000 * 10); // 10 second timeout

    /**
     * Constructor
     *
     * @param nativeMemoryCircuitBreakerMonitorDto contains necessary initialization values
     */
    public NativeMemoryCircuitBreakerMonitor(NativeMemoryCircuitBreakerMonitorDto nativeMemoryCircuitBreakerMonitorDto) {
        this.nativeMemoryCircuitBreaker = nativeMemoryCircuitBreakerMonitorDto.getNativeMemoryCircuitBreaker();
        this.nativeMemoryCacheManager = nativeMemoryCircuitBreakerMonitorDto.getNativeMemoryCacheManager();
        this.clusterService = nativeMemoryCircuitBreakerMonitorDto.getClusterService();
        this.client = nativeMemoryCircuitBreakerMonitorDto.getClient();
        this.threadPool = nativeMemoryCircuitBreakerMonitorDto.getThreadPool();
        this.isStarted = new AtomicReference<>(false);
        this.monitorFuture = null;
    }

    /**
     * Schedules monitor job to be run
     */
    public synchronized void start() {
        // Ensure monitor future is only scheduled once
        boolean isAlreadyStarted = this.isStarted.getAndSet(true);
        if (isAlreadyStarted == false) {
            this.monitorFuture = threadPool.scheduleWithFixedDelay(
                this::monitor,
                TimeValue.timeValueSeconds(CB_TIME_INTERVAL),
                ThreadPool.Names.GENERIC
            );
        }
    }

    @Override
    public synchronized void close() {
        if (this.monitorFuture != null && this.monitorFuture.isCancelled() == false) {
            this.monitorFuture.cancel();
        }
    }

    @VisibleForTesting
    void monitor() {
        if (nativeMemoryCacheManager.isCacheCapacityReached() && clusterService.localNode().isDataNode()) {
            long currentSizeKiloBytes = nativeMemoryCacheManager.getCacheSizeInKilobytes();
            long circuitBreakerLimitSizeKiloBytes = nativeMemoryCircuitBreaker.getLimit().getKb();
            long circuitBreakerUnsetSizeKiloBytes = (long) ((nativeMemoryCircuitBreaker.getUnsetPercentage() / 100)
                * circuitBreakerLimitSizeKiloBytes);
            // Unset capacityReached flag if currentSizeBytes is less than circuitBreakerUnsetSizeBytes
            if (currentSizeKiloBytes <= circuitBreakerUnsetSizeKiloBytes) {
                nativeMemoryCacheManager.setCacheCapacityReached(false);
            }
        }

        // Leader node untriggers CB if all nodes have not reached their max capacity
        if (nativeMemoryCircuitBreaker.isTripped() && clusterService.state().nodes().isLocalNodeElectedClusterManager()) {
            KNNStatsRequest knnStatsRequest = new KNNStatsRequest();
            knnStatsRequest.addStat(StatNames.CACHE_CAPACITY_REACHED.getName());
            knnStatsRequest.timeout(STATS_REQUEST_TIMEOUT);

            try {
                KNNStatsResponse knnStatsResponse = client.execute(KNNStatsAction.INSTANCE, knnStatsRequest).get();
                List<KNNStatsNodeResponse> nodeResponses = knnStatsResponse.getNodes();

                List<String> nodesAtMaxCapacity = new ArrayList<>();
                for (KNNStatsNodeResponse nodeResponse : nodeResponses) {
                    if ((Boolean) nodeResponse.getStatsMap().get(StatNames.CACHE_CAPACITY_REACHED.getName())) {
                        nodesAtMaxCapacity.add(nodeResponse.getNode().getId());
                    }
                }

                if (nodesAtMaxCapacity.isEmpty() == false) {
                    log.info(
                        "[KNN] knn.circuit_breaker.triggered stays set. Nodes at max cache capacity: "
                            + String.join(",", nodesAtMaxCapacity)
                            + "."
                    );
                } else {
                    log.info(
                        "[KNN] Cache capacity below {}% of the circuit breaker limit for all nodes. Unsetting knn.circuit_breaker.triggered flag.",
                        nativeMemoryCircuitBreaker.getUnsetPercentage()
                    );
                    nativeMemoryCircuitBreaker.set(false);
                }
            } catch (Exception e) {
                log.error("[KNN] Error when trying to update the circuit breaker setting", e);
            }
        }
    }
}
