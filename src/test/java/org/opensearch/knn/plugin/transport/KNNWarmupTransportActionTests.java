/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.Version;
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
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.warmup.WarmupSkipReason;
import org.opensearch.core.action.support.DefaultShardOperationFailedException;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
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

        KNNWarmupShardResult result = knnWarmupTransportAction.shardOperation(knnWarmupRequest, shardRouting);
        assertFalse(result.isSkipped());
        assertNull(result.getSkipReason());
        assertEquals(0, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());

        addKnnDoc(testIndexName, "1", testFieldName, new Long[] { 0L, 1L });

        result = knnWarmupTransportAction.shardOperation(knnWarmupRequest, shardRouting);
        assertFalse(result.isSkipped());
        assertEquals(1, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());
    }

    public void testShardOperation_luceneEngineIndex_returnsSkippedResult() throws IOException, ExecutionException, InterruptedException {
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);

        IndexService indexService = createIndex(testIndexName, getKNNDefaultIndexSettingsBuildsGraphAlways());
        createKnnIndexMapping(testIndexName, testFieldName, dimensions, KNNEngine.LUCENE);
        addKnnDoc(testIndexName, "1", testFieldName, new Long[] { 0L, 1L });
        ShardRouting shardRouting = indexService.iterator().next().routingEntry();

        KNNWarmupShardResult result = knnWarmupTransportAction.shardOperation(knnWarmupRequest, shardRouting);
        assertTrue(result.isSkipped());
        assertEquals(WarmupSkipReason.LUCENE_ENGINE, result.getSkipReason());
        assertEquals(0, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());
    }

    public void testNewResponse_aggregatesSkipStatus() {
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);
        List<DefaultShardOperationFailedException> shardFailures = Collections.emptyList();

        KNNWarmupResponse response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            3,
            3,
            0,
            Arrays.asList(KNNWarmupShardResult.executed(), KNNWarmupShardResult.executed(), KNNWarmupShardResult.executed()),
            shardFailures,
            null
        );
        assertEquals(3, response.getTotalShards());
        assertEquals(3, response.getSuccessfulShards());
        assertEquals(0, response.getSkippedShards());
        assertTrue(response.getSkipReasons().isEmpty());

        response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            3,
            3,
            0,
            Arrays.asList(
                KNNWarmupShardResult.skipped(WarmupSkipReason.WARM_TIER_INDEX),
                KNNWarmupShardResult.skipped(WarmupSkipReason.WARM_TIER_INDEX),
                KNNWarmupShardResult.executed()
            ),
            shardFailures,
            null
        );
        assertEquals(2, response.getSkippedShards());
        assertEquals(Collections.singletonList("warm_tier_index"), response.getSkipReasons());

        response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            4,
            4,
            0,
            Arrays.asList(
                KNNWarmupShardResult.skipped(WarmupSkipReason.WARM_TIER_INDEX),
                KNNWarmupShardResult.skipped(WarmupSkipReason.LUCENE_ENGINE),
                KNNWarmupShardResult.skipped(WarmupSkipReason.WARM_TIER_INDEX),
                KNNWarmupShardResult.executed()
            ),
            shardFailures,
            null
        );
        assertEquals(3, response.getSkippedShards());
        assertEquals(Arrays.asList("warm_tier_index", "lucene_engine"), response.getSkipReasons());
    }

    public void testShardResultSerialization() throws IOException {
        KNNWarmupShardResult executed = KNNWarmupShardResult.executed();
        BytesStreamOutput out = new BytesStreamOutput();
        executed.writeTo(out);
        KNNWarmupShardResult deserialized = new KNNWarmupShardResult(out.bytes().streamInput());
        assertFalse(deserialized.isSkipped());
        assertNull(deserialized.getSkipReason());

        KNNWarmupShardResult skipped = KNNWarmupShardResult.skipped(WarmupSkipReason.LUCENE_ENGINE);
        out = new BytesStreamOutput();
        skipped.writeTo(out);
        deserialized = new KNNWarmupShardResult(out.bytes().streamInput());
        assertTrue(deserialized.isSkipped());
        assertEquals(WarmupSkipReason.LUCENE_ENGINE, deserialized.getSkipReason());

        // Before 3.9.0 the shard result carried no information: nothing is written and the read
        // result defaults to executed
        out = new BytesStreamOutput();
        out.setVersion(Version.V_3_8_0);
        skipped.writeTo(out);
        assertEquals(0, out.bytes().length());
        try (StreamInput in = out.bytes().streamInput()) {
            in.setVersion(Version.V_3_8_0);
            deserialized = new KNNWarmupShardResult(in);
        }
        assertFalse(deserialized.isSkipped());
        assertNull(deserialized.getSkipReason());

        // An unknown skip reason (e.g. written by a newer node during a rolling upgrade) must not
        // fail deserialization and resolves to UNKNOWN
        out = new BytesStreamOutput();
        out.writeBoolean(true);
        out.writeString("some_future_reason");
        try (StreamInput in = out.bytes().streamInput()) {
            deserialized = new KNNWarmupShardResult(in);
        }
        assertTrue(deserialized.isSkipped());
        assertEquals(WarmupSkipReason.UNKNOWN, deserialized.getSkipReason());
    }

    public void testResponseSerialization() throws IOException {
        List<String> skipReasons = Arrays.asList("warm_tier_index", "lucene_engine");
        KNNWarmupResponse original = new KNNWarmupResponse(10, 10, 0, Collections.emptyList(), 5, skipReasons);

        BytesStreamOutput out = new BytesStreamOutput();
        original.writeTo(out);
        KNNWarmupResponse deserialized = new KNNWarmupResponse(out.bytes().streamInput());
        assertEquals(original.getTotalShards(), deserialized.getTotalShards());
        assertEquals(original.getSuccessfulShards(), deserialized.getSuccessfulShards());
        assertEquals(original.getFailedShards(), deserialized.getFailedShards());
        assertEquals(original.getSkippedShards(), deserialized.getSkippedShards());
        assertEquals(original.getSkipReasons(), deserialized.getSkipReasons());

        // Before 3.9.0 the skip fields are not written and default to zero / empty on read
        out = new BytesStreamOutput();
        out.setVersion(Version.V_3_8_0);
        original.writeTo(out);
        try (StreamInput in = out.bytes().streamInput()) {
            in.setVersion(Version.V_3_8_0);
            deserialized = new KNNWarmupResponse(in);
        }
        assertEquals(10, deserialized.getTotalShards());
        assertEquals(10, deserialized.getSuccessfulShards());
        assertEquals(0, deserialized.getSkippedShards());
        assertTrue(deserialized.getSkipReasons().isEmpty());
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
