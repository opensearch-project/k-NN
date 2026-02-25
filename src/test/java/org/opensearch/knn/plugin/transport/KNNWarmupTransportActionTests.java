/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlock;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.block.ClusterBlocks;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.cluster.routing.ShardsIterator;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.index.IndexService;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.core.rest.RestStatus;

import java.io.IOException;
import java.util.EnumSet;
import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNWarmupTransportActionTests extends KNNSingleNodeTestCase {
    private final String testIndexName = "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 2;

    public void testShardOperation() throws IOException, ExecutionException, InterruptedException {
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);
        IndexService indexService;
        ShardRouting shardRouting;
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        assertEquals(0, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());

        indexService = createIndex(testIndexName, getKNNDefaultIndexSettingsBuildsGraphAlways());
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);
        shardRouting = indexService.iterator().next().routingEntry();

        knnWarmupTransportAction.shardOperation(knnWarmupRequest, shardRouting);
        assertEquals(0, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());

        addKnnDoc(testIndexName, "1", testFieldName, new Long[] { 0L, 1L });

        knnWarmupTransportAction.shardOperation(knnWarmupRequest, shardRouting);
        assertEquals(1, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());
    }

    public void testShards() throws InterruptedException, ExecutionException, IOException {
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);

        createKNNIndex(testIndexName);
        createKnnIndexMapping(testIndexName, testFieldName, dimensions);
        addKnnDoc(testIndexName, "1", testFieldName, new Long[] { 0L, 1L });

        ShardsIterator shardsIterator = knnWarmupTransportAction.shards(
            clusterService.state(),
            knnWarmupRequest,
            new String[] { testIndexName }
        );
        assertEquals(1, shardsIterator.size());
    }

    public void testCheckGlobalBlock() {
        ClusterService clusterService = mock(ClusterService.class);
        ClusterBlock metaReadClusterBlock = new ClusterBlock(
            randomInt(),
            "test-meta-data-block",
            false,
            false,
            false,
            RestStatus.FORBIDDEN,
            EnumSet.of(ClusterBlockLevel.METADATA_READ)
        );
        ClusterBlocks clusterBlocks = ClusterBlocks.builder().addGlobalBlock(metaReadClusterBlock).build();
        ClusterState state = ClusterState.builder(ClusterName.DEFAULT).blocks(clusterBlocks).build();
        when(clusterService.state()).thenReturn(state);

        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);
        assertNotNull(knnWarmupTransportAction.checkGlobalBlock(clusterService.state(), knnWarmupRequest));
    }

    public void testCheckRequestBlock() {
        ClusterService clusterService = mock(ClusterService.class);
        ClusterBlock metaReadClusterBlock = new ClusterBlock(
            randomInt(),
            "test-meta-data-block",
            false,
            false,
            false,
            RestStatus.FORBIDDEN,
            EnumSet.of(ClusterBlockLevel.METADATA_READ)
        );
        ClusterBlocks clusterBlocks = ClusterBlocks.builder().addGlobalBlock(metaReadClusterBlock).build();
        ClusterState state = ClusterState.builder(ClusterName.DEFAULT).blocks(clusterBlocks).build();
        when(clusterService.state()).thenReturn(state);

        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);
        assertNotNull(knnWarmupTransportAction.checkRequestBlock(clusterService.state(), knnWarmupRequest, new String[] { testIndexName }));
    }
}
