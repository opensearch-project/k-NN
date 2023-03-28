/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory.breaker;

import lombok.Builder;
import lombok.Value;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.threadpool.ThreadPool;

@Value
@Builder
public class NativeMemoryCircuitBreakerMonitorDto {
    NativeMemoryCircuitBreaker nativeMemoryCircuitBreaker;
    NativeMemoryCacheManager nativeMemoryCacheManager;
    ClusterService clusterService;
    Client client;
    ThreadPool threadPool;
}
