/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory.breaker;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.opensearch.action.support.PlainActionFuture;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.ByteSizeUnit;
import org.opensearch.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsNodeResponse;
import org.opensearch.knn.plugin.transport.KNNStatsRequest;
import org.opensearch.knn.plugin.transport.KNNStatsResponse;
import org.opensearch.threadpool.Scheduler;
import org.opensearch.threadpool.ThreadPool;

import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED;
import static org.opensearch.knn.index.KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;
import static org.opensearch.knn.index.KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT;

public class NativeMemoryCircuitBreakerServiceTests extends KNNTestCase {

    public void testSetCircuitBreaker() {
        boolean isTriggered = randomBoolean();
        doNothing().when(knnSettings).updateBooleanSetting(KNN_CIRCUIT_BREAKER_TRIGGERED, isTriggered);
        NativeMemoryCircuitBreakerService nativeMemoryCircuitBreakerService = new NativeMemoryCircuitBreakerService(
            knnSettings,
            threadPool,
            clusterService,
            client
        );
        nativeMemoryCircuitBreakerService.setCircuitBreaker(isTriggered);
        verify(knnSettings, times(1)).updateBooleanSetting(KNN_CIRCUIT_BREAKER_TRIGGERED, isTriggered);
    }

    public void testGetCircuitBreakerLimit() {
        ByteSizeValue circuitBreakerLimit = new ByteSizeValue(randomIntBetween(10, 10000), ByteSizeUnit.KB);
        when(knnSettings.getSettingValue(KNN_MEMORY_CIRCUIT_BREAKER_LIMIT)).thenReturn(circuitBreakerLimit);
        NativeMemoryCircuitBreakerService nativeMemoryCircuitBreakerService = new NativeMemoryCircuitBreakerService(
            knnSettings,
            threadPool,
            clusterService,
            client
        );
        assertEquals(circuitBreakerLimit, nativeMemoryCircuitBreakerService.getCircuitBreakerLimit());
    }

    public void testIsCircuitBreakerTriggered() {
        boolean isTriggered = randomBoolean();
        when(knnSettings.getSettingValue(KNN_CIRCUIT_BREAKER_TRIGGERED)).thenReturn(isTriggered);
        NativeMemoryCircuitBreakerService nativeMemoryCircuitBreakerService = new NativeMemoryCircuitBreakerService(
            knnSettings,
            threadPool,
            clusterService,
            client
        );
        assertEquals(isTriggered, nativeMemoryCircuitBreakerService.isCircuitBreakerTriggered());
    }

    public void testIsCircuitBreakerEnabled() {
        boolean isEnabled = randomBoolean();
        when(knnSettings.getSettingValue(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED)).thenReturn(isEnabled);
        NativeMemoryCircuitBreakerService nativeMemoryCircuitBreakerService = new NativeMemoryCircuitBreakerService(
            knnSettings,
            threadPool,
            clusterService,
            client
        );
        assertEquals(isEnabled, nativeMemoryCircuitBreakerService.isCircuitBreakerEnabled());
    }

    public void testStartStopLifeCycle() {
        Scheduler.Cancellable cancellable = createCancellable();
        when(threadPool.scheduleWithFixedDelay(any(), any(), any())).thenReturn(cancellable);
        NativeMemoryCircuitBreakerService nativeMemoryCircuitBreakerService = createNativeMemoryCircuitBreakerServiceWithNoopMonitor(
            knnSettings,
            threadPool,
            clusterService,
            client
        );
        nativeMemoryCircuitBreakerService.doStart();
        assertEquals(cancellable, nativeMemoryCircuitBreakerService.circuitBreakerFuture.get());
        assertFalse(nativeMemoryCircuitBreakerService.circuitBreakerFuture.get().isCancelled());
        nativeMemoryCircuitBreakerService.doStop();
        assertNull(nativeMemoryCircuitBreakerService.circuitBreakerFuture.get());
    }

    public void testStartCloseLifeCycle() {
        Scheduler.Cancellable cancellable = createCancellable();
        when(threadPool.scheduleWithFixedDelay(any(), any(), any())).thenReturn(cancellable);
        NativeMemoryCircuitBreakerService nativeMemoryCircuitBreakerService = createNativeMemoryCircuitBreakerServiceWithNoopMonitor(
            knnSettings,
            threadPool,
            clusterService,
            client
        );
        nativeMemoryCircuitBreakerService.doStart();
        assertEquals(cancellable, nativeMemoryCircuitBreakerService.circuitBreakerFuture.get());
        assertFalse(nativeMemoryCircuitBreakerService.circuitBreakerFuture.get().isCancelled());
        nativeMemoryCircuitBreakerService.doClose();
        assertEquals(cancellable, nativeMemoryCircuitBreakerService.circuitBreakerFuture.get());
        assertTrue(nativeMemoryCircuitBreakerService.circuitBreakerFuture.get().isCancelled());
    }

    public void testMonitorRun_whenDataNodeCacheSizeLowerThanThreshold_thenUnsetCacheCapacityReached() {
        // Setup state so that cache capacity is marked as reached but the ratio in the cache is less than
        // the unset ratio
        when(nativeMemoryCacheManager.isCacheCapacityReached()).thenReturn(true);
        when(node.isDataNode()).thenReturn(true);
        when(clusterService.localNode()).thenReturn(node);
        long cacheSizeInKb = 1;
        long cbLimitInKb = 100;
        double unsetSizeInKb = 99;
        when(nativeMemoryCacheManager.getCacheSizeInKilobytes()).thenReturn(cacheSizeInKb);
        when(nativeMemoryCircuitBreakerService.getCircuitBreakerLimit()).thenReturn(new ByteSizeValue(cbLimitInKb, ByteSizeUnit.KB));
        when(nativeMemoryCircuitBreakerService.getCircuitBreakerUnsetPercentage()).thenReturn(unsetSizeInKb);

        // Avoid duties of cluster manager
        when(nativeMemoryCircuitBreakerService.isCircuitBreakerTriggered()).thenReturn(false);

        doNothing().when(nativeMemoryCacheManager).setCacheCapacityReached(false);
        NativeMemoryCircuitBreakerService.Monitor monitor = new NativeMemoryCircuitBreakerService.Monitor(
            nativeMemoryCircuitBreakerService,
            nativeMemoryCacheManager,
            clusterService,
            client
        );
        monitor.run();
        verify(nativeMemoryCacheManager, times(1)).setCacheCapacityReached(false);
    }

    @SneakyThrows
    public void testMonitorRun_whenClusterManagerAndClusterHasCapacity_thenUnsetCircuitBreaker() {
        // Setup state so that current node is cluster manager and should unset circuit breaker5
        when(nativeMemoryCircuitBreakerService.isCircuitBreakerTriggered()).thenReturn(true);
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

        doNothing().when(nativeMemoryCircuitBreakerService).setCircuitBreaker(false);
        NativeMemoryCircuitBreakerService.Monitor monitor = new NativeMemoryCircuitBreakerService.Monitor(
            nativeMemoryCircuitBreakerService,
            nativeMemoryCacheManager,
            clusterService,
            client
        );
        monitor.run();
        verify(nativeMemoryCircuitBreakerService, times(1)).setCircuitBreaker(false);
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

    private NativeMemoryCircuitBreakerService createNativeMemoryCircuitBreakerServiceWithNoopMonitor(
        KNNSettings knnSettings,
        ThreadPool threadPool,
        ClusterService clusterService,
        Client client
    ) {
        return new NativeMemoryCircuitBreakerService(knnSettings, threadPool, clusterService, client) {
            @Override
            protected Monitor getMonitor() {
                return mock(Monitor.class);
            }
        };
    }
}
