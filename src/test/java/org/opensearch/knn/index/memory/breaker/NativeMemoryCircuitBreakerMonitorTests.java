/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory.breaker;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.opensearch.action.support.PlainActionFuture;
import org.opensearch.common.unit.ByteSizeUnit;
import org.opensearch.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsNodeResponse;
import org.opensearch.knn.plugin.transport.KNNStatsRequest;
import org.opensearch.knn.plugin.transport.KNNStatsResponse;
import org.opensearch.threadpool.Scheduler;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class NativeMemoryCircuitBreakerMonitorTests extends KNNTestCase {

    public void testStart_whenCalledMultipleTimes_thenScheduleMonitorOnce() {
        Scheduler.Cancellable cancellable = createCancellable();
        when(threadPool.scheduleWithFixedDelay(any(), any(), any())).thenReturn(cancellable);
        NativeMemoryCircuitBreakerMonitor monitor = new NativeMemoryCircuitBreakerMonitor(
            NativeMemoryCircuitBreakerMonitorDto.builder()
                .nativeMemoryCacheManager(nativeMemoryCacheManager)
                .nativeMemoryCircuitBreaker(nativeMemoryCircuitBreaker)
                .threadPool(threadPool)
                .client(client)
                .clusterService(clusterService)
                .build()
        );

        monitor.start();
        monitor.start();
        monitor.start();

        verify(threadPool, times(1)).scheduleWithFixedDelay(any(), any(), any());
    }

    public void testClose_whenCalled_thenCancel() {
        Scheduler.Cancellable cancellable = createCancellable();
        when(threadPool.scheduleWithFixedDelay(any(), any(), any())).thenReturn(cancellable);
        NativeMemoryCircuitBreakerMonitor monitor = new NativeMemoryCircuitBreakerMonitor(
            NativeMemoryCircuitBreakerMonitorDto.builder()
                .nativeMemoryCacheManager(nativeMemoryCacheManager)
                .nativeMemoryCircuitBreaker(nativeMemoryCircuitBreaker)
                .threadPool(threadPool)
                .client(client)
                .clusterService(clusterService)
                .build()
        );

        monitor.start();
        assertFalse(cancellable.isCancelled());
        monitor.close();
        assertTrue(cancellable.isCancelled());
    }

    public void testMonitor_whenDataNodeCacheSizeLowerThanThreshold_thenUnsetCacheCapacityReached() {
        // Setup state so that cache capacity is marked as reached but the ratio in the cache is less than
        // the unset ratio
        when(nativeMemoryCacheManager.isCacheCapacityReached()).thenReturn(true);
        when(node.isDataNode()).thenReturn(true);
        when(clusterService.localNode()).thenReturn(node);
        long cacheSizeInKb = 1;
        long cbLimitInKb = 100;
        double unsetSizeInKb = 99;
        when(nativeMemoryCacheManager.getCacheSizeInKilobytes()).thenReturn(cacheSizeInKb);
        when(nativeMemoryCircuitBreaker.getLimit()).thenReturn(new ByteSizeValue(cbLimitInKb, ByteSizeUnit.KB));
        when(nativeMemoryCircuitBreaker.getUnsetPercentage()).thenReturn(unsetSizeInKb);

        // Avoid duties of cluster manager
        when(nativeMemoryCircuitBreaker.isTripped()).thenReturn(false);

        doNothing().when(nativeMemoryCacheManager).setCacheCapacityReached(false);
        NativeMemoryCircuitBreakerMonitor monitor = new NativeMemoryCircuitBreakerMonitor(
            NativeMemoryCircuitBreakerMonitorDto.builder()
                .nativeMemoryCacheManager(nativeMemoryCacheManager)
                .nativeMemoryCircuitBreaker(nativeMemoryCircuitBreaker)
                .threadPool(threadPool)
                .client(client)
                .clusterService(clusterService)
                .build()
        );

        monitor.monitor();
        verify(nativeMemoryCacheManager, times(1)).setCacheCapacityReached(false);
    }

    @SneakyThrows
    public void testMonitorRun_whenClusterManagerAndClusterHasCapacity_thenUnsetCircuitBreaker() {
        // Setup state so that current node is cluster manager and should unset circuit breaker5
        when(nativeMemoryCircuitBreaker.isTripped()).thenReturn(true);
        when(discoveryNodes.isLocalNodeElectedClusterManager()).thenReturn(true);
        when(clusterState.nodes()).thenReturn(discoveryNodes);
        when(clusterService.state()).thenReturn(clusterState);

        // Ensure all nodes have cache capacity as not reached
        Map<String, Object> reachedMap = ImmutableMap.of(StatNames.CACHE_CAPACITY_REACHED.getName(), false);
        List<KNNStatsNodeResponse> nodeResponses = List.of(
            new KNNStatsNodeResponse(node, reachedMap),
            new KNNStatsNodeResponse(node, reachedMap),
            new KNNStatsNodeResponse(node, reachedMap)
        );
        KNNStatsResponse knnStatsResponse = mock(KNNStatsResponse.class);
        when(knnStatsResponse.getNodes()).thenReturn(nodeResponses);

        PlainActionFuture<KNNStatsResponse> actionFuture = new PlainActionFuture<>() {
            @Override
            public KNNStatsResponse get() {
                return knnStatsResponse;
            }
        };

        when(client.execute(any(KNNStatsAction.class), any(KNNStatsRequest.class))).thenReturn(actionFuture);

        // Avoid duties of data node
        when(nativeMemoryCacheManager.isCacheCapacityReached()).thenReturn(false);

        doNothing().when(nativeMemoryCircuitBreaker).set(false);
        NativeMemoryCircuitBreakerMonitor monitor = new NativeMemoryCircuitBreakerMonitor(
            NativeMemoryCircuitBreakerMonitorDto.builder()
                .nativeMemoryCacheManager(nativeMemoryCacheManager)
                .nativeMemoryCircuitBreaker(nativeMemoryCircuitBreaker)
                .threadPool(threadPool)
                .client(client)
                .clusterService(clusterService)
                .build()
        );
        monitor.monitor();
        verify(nativeMemoryCircuitBreaker, times(1)).set(false);
    }

    private Scheduler.Cancellable createCancellable() {
        return new Scheduler.Cancellable() {
            boolean isCancelled = false;

            @Override
            public boolean cancel() {
                isCancelled = true;
                return true;
            }

            @Override
            public boolean isCancelled() {
                return isCancelled;
            }
        };
    }

}
