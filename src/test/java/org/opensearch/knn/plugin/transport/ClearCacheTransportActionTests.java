/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.SneakyThrows;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlockException;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.cluster.routing.ShardsIterator;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.index.IndexService;
import org.opensearch.knn.KNNCommonSettingsBuilder;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;

import java.util.EnumSet;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class ClearCacheTransportActionTests extends KNNSingleNodeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 2;

    @SneakyThrows
    public void testShardOperation() {
        String testIndex = getTestName().toLowerCase();
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndex);
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        assertEquals(0, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());

        IndexService indexService = createIndex(testIndex, KNNCommonSettingsBuilder.defaultSettings().build());
        createKnnIndexMapping(testIndex, TEST_FIELD, DIMENSIONS);
        addKnnDoc(testIndex, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });
        ShardRouting shardRouting = indexService.iterator().next().routingEntry();

        knnWarmupTransportAction.shardOperation(knnWarmupRequest, shardRouting);
        assertEquals(1, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());

        ClearCacheRequest clearCacheRequest = new ClearCacheRequest(testIndex);
        ClearCacheTransportAction clearCacheTransportAction = node().injector().getInstance(ClearCacheTransportAction.class);
        clearCacheTransportAction.shardOperation(clearCacheRequest, shardRouting);
        assertEquals(0, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());
    }

    @SneakyThrows
    public void testShards() {
        String testIndex = getTestName().toLowerCase();
        ClusterService clusterService = node().injector().getInstance(ClusterService.class);
        ClearCacheTransportAction clearCacheTransportAction = node().injector().getInstance(ClearCacheTransportAction.class);
        ClearCacheRequest clearCacheRequest = new ClearCacheRequest(testIndex);

        createIndex(testIndex, KNNCommonSettingsBuilder.defaultSettings().build());
        createKnnIndexMapping(testIndex, TEST_FIELD, DIMENSIONS);
        addKnnDoc(testIndex, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        ShardsIterator shardsIterator = clearCacheTransportAction.shards(
            clusterService.state(),
            clearCacheRequest,
            new String[] { testIndex }
        );
        assertEquals(1, shardsIterator.size());
    }

    public void testCheckGlobalBlock_throwsClusterBlockException() {
        String testIndex = getTestName().toLowerCase();
        String description = "testing metadata block";
        ClusterService clusterService = mock(ClusterService.class);
        addGlobalClusterBlock(clusterService, description, EnumSet.of(ClusterBlockLevel.METADATA_WRITE));
        ClearCacheTransportAction clearCacheTransportAction = node().injector().getInstance(ClearCacheTransportAction.class);
        ClearCacheRequest clearCacheRequest = new ClearCacheRequest(testIndex);
        ClusterBlockException ex = clearCacheTransportAction.checkGlobalBlock(clusterService.state(), clearCacheRequest);
        assertTrue(ex.getMessage().contains(description));
    }

    public void testCheckGlobalBlock_notThrowsClusterBlockException() {
        String testIndex = getTestName().toLowerCase();
        ClusterService clusterService = mock(ClusterService.class);
        ClearCacheTransportAction clearCacheTransportAction = node().injector().getInstance(ClearCacheTransportAction.class);
        ClearCacheRequest clearCacheRequest = new ClearCacheRequest(testIndex);
        ClusterState state = ClusterState.builder(ClusterName.DEFAULT).build();
        when(clusterService.state()).thenReturn(state);
        assertNull(clearCacheTransportAction.checkGlobalBlock(clusterService.state(), clearCacheRequest));
    }

    public void testCheckRequestBlock_throwsClusterBlockException() {
        String testIndex = getTestName().toLowerCase();
        String description = "testing index metadata block";
        ClusterService clusterService = mock(ClusterService.class);
        addIndexClusterBlock(clusterService, description, EnumSet.of(ClusterBlockLevel.METADATA_WRITE), testIndex);

        ClearCacheTransportAction clearCacheTransportAction = node().injector().getInstance(ClearCacheTransportAction.class);
        ClearCacheRequest clearCacheRequest = new ClearCacheRequest(testIndex);
        ClusterBlockException ex = clearCacheTransportAction.checkRequestBlock(
            clusterService.state(),
            clearCacheRequest,
            new String[] { testIndex }
        );
        assertTrue(ex.getMessage().contains(testIndex));
        assertTrue(ex.getMessage().contains(description));

    }

    public void testCheckRequestBlock_notThrowsClusterBlockException() {
        String testIndex = getTestName().toLowerCase();
        ClusterService clusterService = mock(ClusterService.class);
        ClearCacheTransportAction clearCacheTransportAction = node().injector().getInstance(ClearCacheTransportAction.class);
        ClearCacheRequest clearCacheRequest = new ClearCacheRequest(testIndex);
        ClusterState state = ClusterState.builder(ClusterName.DEFAULT).build();
        when(clusterService.state()).thenReturn(state);
        assertNull(clearCacheTransportAction.checkRequestBlock(clusterService.state(), clearCacheRequest, new String[] { testIndex }));
    }
}
