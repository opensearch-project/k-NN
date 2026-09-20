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
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
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
import java.util.Map;
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
        assertEquals(testIndexName, result.getIndexName());
        assertTrue(result.getSkippedFields().isEmpty());
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
        assertEquals(WarmupSkipReason.NO_OFF_HEAP_WARMUP, result.getSkipReason());
        assertEquals(testIndexName, result.getIndexName());
        assertEquals(Collections.singletonList(testFieldName), result.getSkippedFields());
        assertEquals(0, NativeMemoryCacheManager.getInstance().getIndicesCacheStats().size());
    }

    public void testNewResponse_aggregatesSkipStatus() {
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        createIndex(testIndexName, indexSettings(3, 0));
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);
        List<DefaultShardOperationFailedException> shardFailures = Collections.emptyList();

        KNNWarmupResponse response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            3,
            3,
            0,
            Arrays.asList(
                KNNWarmupShardResult.executed(testIndexName, Collections.emptyList()),
                KNNWarmupShardResult.executed(testIndexName, Collections.emptyList()),
                KNNWarmupShardResult.executed(testIndexName, Collections.emptyList())
            ),
            shardFailures,
            clusterState()
        );
        assertEquals(3, response.getTotalShards());
        assertEquals(3, response.getSuccessfulShards());
        assertEquals(0, response.getSkippedShards());
        assertEquals(1, response.getIndexStats().size());
        assertEquals(testIndexName, response.getIndexStats().get(0).getIndexName());
        assertEquals(3, response.getIndexStats().get(0).getTotalShards());
        assertEquals(3, response.getIndexStats().get(0).getSuccessfulShards());
        assertEquals(0, response.getIndexStats().get(0).getSkippedShards());
        assertTrue(response.getIndexStats().get(0).getSkipReasons().isEmpty());
        assertTrue(response.getIndexStats().get(0).getSkippedFields().isEmpty());

        // Mixed: 2 skipped with the same reason, 1 executed
        response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            3,
            3,
            0,
            Arrays.asList(
                KNNWarmupShardResult.skipped(testIndexName, WarmupSkipReason.WARM_TIER_INDEX, Collections.emptyList()),
                KNNWarmupShardResult.skipped(testIndexName, WarmupSkipReason.WARM_TIER_INDEX, Collections.emptyList()),
                KNNWarmupShardResult.executed(testIndexName, Collections.emptyList())
            ),
            shardFailures,
            clusterState()
        );
        assertEquals(2, response.getSkippedShards());
        assertEquals(2, response.getIndexStats().get(0).getSkippedShards());
        assertEquals(List.of(WarmupSkipReason.WARM_TIER_INDEX), response.getIndexStats().get(0).getSkipReasons());

        // Different reasons within one index are all reported, without duplicates
        response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            3,
            3,
            0,
            Arrays.asList(
                KNNWarmupShardResult.skipped(testIndexName, WarmupSkipReason.NO_OFF_HEAP_WARMUP, Collections.emptyList()),
                KNNWarmupShardResult.skipped(testIndexName, WarmupSkipReason.UNKNOWN, Collections.emptyList()),
                KNNWarmupShardResult.skipped(testIndexName, WarmupSkipReason.NO_OFF_HEAP_WARMUP, Collections.emptyList())
            ),
            shardFailures,
            clusterState()
        );
        assertEquals(
            List.of(WarmupSkipReason.NO_OFF_HEAP_WARMUP, WarmupSkipReason.UNKNOWN),
            response.getIndexStats().get(0).getSkipReasons()
        );
    }

    public void testNewResponse_aggregatesPerIndex() {
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        createIndex("warm-index", indexSettings(2, 0));
        createIndex("lucene-index", indexSettings(2, 0));
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest("warm-index", "lucene-index");

        KNNWarmupResponse response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            4,
            4,
            0,
            Arrays.asList(
                KNNWarmupShardResult.skipped("warm-index", WarmupSkipReason.WARM_TIER_INDEX, Collections.emptyList()),
                KNNWarmupShardResult.skipped(
                    "lucene-index",
                    WarmupSkipReason.NO_OFF_HEAP_WARMUP,
                    Collections.singletonList("lucene-field")
                ),
                KNNWarmupShardResult.skipped("warm-index", WarmupSkipReason.WARM_TIER_INDEX, Collections.emptyList()),
                KNNWarmupShardResult.executed("lucene-index", Collections.singletonList("lucene-field"))
            ),
            Collections.emptyList(),
            clusterState()
        );
        assertEquals(3, response.getSkippedShards());
        assertEquals(2, response.getIndexStats().size());

        KNNWarmupResponse.IndexWarmupStats warmIndexStats = getIndexStats(response, "warm-index");
        assertEquals(2, warmIndexStats.getTotalShards());
        assertEquals(2, warmIndexStats.getSkippedShards());
        assertEquals(List.of(WarmupSkipReason.WARM_TIER_INDEX), warmIndexStats.getSkipReasons());

        KNNWarmupResponse.IndexWarmupStats luceneIndexStats = getIndexStats(response, "lucene-index");
        assertEquals(2, luceneIndexStats.getTotalShards());
        assertEquals(1, luceneIndexStats.getSkippedShards());
        assertEquals(List.of(WarmupSkipReason.NO_OFF_HEAP_WARMUP), luceneIndexStats.getSkipReasons());
        // Field names are deduplicated across the shards of the index
        assertEquals(Collections.singletonList("lucene-field"), luceneIndexStats.getSkippedFields());
    }

    public void testNewResponse_aggregatesFailuresPerIndex() {
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        createIndex(testIndexName, indexSettings(3, 0));
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);

        KNNWarmupResponse response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            3,
            2,
            1,
            Arrays.asList(
                KNNWarmupShardResult.executed(testIndexName, Collections.emptyList()),
                KNNWarmupShardResult.executed(testIndexName, Collections.emptyList())
            ),
            Collections.singletonList(new DefaultShardOperationFailedException(testIndexName, 0, new RuntimeException("boom"))),
            clusterState()
        );
        assertEquals(1, response.getFailedShards());
        assertEquals(1, response.getIndexStats().size());
        assertEquals(testIndexName, response.getIndexStats().get(0).getIndexName());
        assertEquals(3, response.getIndexStats().get(0).getTotalShards());
        assertEquals(2, response.getIndexStats().get(0).getSuccessfulShards());
        assertEquals(1, response.getIndexStats().get(0).getFailedShards());
    }

    public void testNewResponse_countsUnavailableShardsInIndexTotal() {
        KNNWarmupTransportAction knnWarmupTransportAction = node().injector().getInstance(KNNWarmupTransportAction.class);
        // The replica cannot be assigned on a single node: it is neither a result nor a failure
        createIndex(testIndexName, indexSettings(1, 1));
        KNNWarmupRequest knnWarmupRequest = new KNNWarmupRequest(testIndexName);

        KNNWarmupResponse response = knnWarmupTransportAction.newResponse(
            knnWarmupRequest,
            2,
            1,
            0,
            Collections.singletonList(KNNWarmupShardResult.executed(testIndexName, Collections.emptyList())),
            Collections.emptyList(),
            clusterState()
        );
        assertEquals(1, response.getIndexStats().size());
        assertEquals(response.getTotalShards(), response.getIndexStats().get(0).getTotalShards());
        assertEquals(1, response.getIndexStats().get(0).getSuccessfulShards());
        assertEquals(0, response.getIndexStats().get(0).getFailedShards());
    }

    @SuppressWarnings("unchecked")
    public void testResponseToXContent() throws IOException {
        List<KNNWarmupResponse.IndexWarmupStats> indexStats = Arrays.asList(
            new KNNWarmupResponse.IndexWarmupStats("warm-index", 1, 1, 1, 0, List.of(WarmupSkipReason.WARM_TIER_INDEX), List.of()),
            new KNNWarmupResponse.IndexWarmupStats("mixed-index", 1, 1, 0, 0, List.of(), List.of("lucene-field"))
        );
        KNNWarmupResponse response = new KNNWarmupResponse(2, 2, 0, Collections.emptyList(), 1, indexStats);

        XContentBuilder builder = XContentFactory.jsonBuilder();
        response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        Map<String, Object> responseMap = XContentHelper.convertToMap(BytesReference.bytes(builder), false, builder.contentType()).v2();

        Map<String, Object> shards = (Map<String, Object>) responseMap.get(KNNWarmupResponse.SHARDS_FIELD);
        assertEquals(2, shards.get(KNNWarmupResponse.TOTAL_FIELD));
        assertEquals(1, shards.get(KNNWarmupResponse.SKIPPED_FIELD));
        assertFalse(shards.containsKey(KNNWarmupResponse.SKIP_REASONS_FIELD));
        assertFalse(shards.containsKey(KNNWarmupResponse.FAILURES_FIELD));

        Map<String, Object> indices = (Map<String, Object>) responseMap.get(KNNWarmupResponse.INDICES_FIELD);
        Map<String, Object> warmIndex = (Map<String, Object>) indices.get("warm-index");
        Map<String, Object> warmIndexShards = (Map<String, Object>) warmIndex.get(KNNWarmupResponse.SHARDS_FIELD);
        assertEquals(1, warmIndexShards.get(KNNWarmupResponse.SKIPPED_FIELD));
        assertFalse(warmIndexShards.containsKey(KNNWarmupResponse.SKIP_REASONS_FIELD));
        assertEquals(List.of("warm_tier_index"), warmIndex.get(KNNWarmupResponse.SKIP_REASONS_FIELD));
        assertFalse(warmIndex.containsKey(KNNWarmupResponse.SKIPPED_FIELDS_FIELD));

        Map<String, Object> mixedIndex = (Map<String, Object>) indices.get("mixed-index");
        assertFalse(mixedIndex.containsKey(KNNWarmupResponse.SKIP_REASONS_FIELD));
        assertEquals(List.of("lucene-field"), mixedIndex.get(KNNWarmupResponse.SKIPPED_FIELDS_FIELD));

        // Without per-index details the indices object is omitted
        builder = XContentFactory.jsonBuilder();
        new KNNWarmupResponse(1, 1, 0, Collections.emptyList()).toXContent(builder, ToXContent.EMPTY_PARAMS);
        responseMap = XContentHelper.convertToMap(BytesReference.bytes(builder), false, builder.contentType()).v2();
        assertFalse(responseMap.containsKey(KNNWarmupResponse.INDICES_FIELD));
    }

    public void testShardResultSerialization() throws IOException {
        KNNWarmupShardResult executed = KNNWarmupShardResult.executed(testIndexName, Collections.singletonList("lucene-field"));
        BytesStreamOutput out = new BytesStreamOutput();
        executed.writeTo(out);
        KNNWarmupShardResult deserialized = new KNNWarmupShardResult(out.bytes().streamInput());
        assertFalse(deserialized.isSkipped());
        assertNull(deserialized.getSkipReason());
        assertEquals(testIndexName, deserialized.getIndexName());
        assertEquals(Collections.singletonList("lucene-field"), deserialized.getSkippedFields());

        KNNWarmupShardResult skipped = KNNWarmupShardResult.skipped(
            testIndexName,
            WarmupSkipReason.NO_OFF_HEAP_WARMUP,
            Collections.emptyList()
        );
        out = new BytesStreamOutput();
        skipped.writeTo(out);
        deserialized = new KNNWarmupShardResult(out.bytes().streamInput());
        assertTrue(deserialized.isSkipped());
        assertEquals(WarmupSkipReason.NO_OFF_HEAP_WARMUP, deserialized.getSkipReason());
        assertEquals(testIndexName, deserialized.getIndexName());
        assertTrue(deserialized.getSkippedFields().isEmpty());

        // Before 3.10.0 the shard result carried no information: nothing is written and the read
        // result defaults to executed with no index attribution
        out = new BytesStreamOutput();
        out.setVersion(Version.V_3_9_0);
        skipped.writeTo(out);
        assertEquals(0, out.bytes().length());
        try (StreamInput in = out.bytes().streamInput()) {
            in.setVersion(Version.V_3_9_0);
            deserialized = new KNNWarmupShardResult(in);
        }
        assertFalse(deserialized.isSkipped());
        assertNull(deserialized.getSkipReason());
        assertNull(deserialized.getIndexName());
        assertTrue(deserialized.getSkippedFields().isEmpty());

        // An unknown skip reason (e.g. written by a newer node during a rolling upgrade) must not
        // fail deserialization and resolves to UNKNOWN
        out = new BytesStreamOutput();
        out.writeBoolean(true);
        out.writeString("some_future_reason");
        out.writeString(testIndexName);
        out.writeStringCollection(List.of());
        try (StreamInput in = out.bytes().streamInput()) {
            deserialized = new KNNWarmupShardResult(in);
        }
        assertTrue(deserialized.isSkipped());
        assertEquals(WarmupSkipReason.UNKNOWN, deserialized.getSkipReason());
    }

    public void testResponseSerialization() throws IOException {
        List<KNNWarmupResponse.IndexWarmupStats> indexStats = Arrays.asList(
            new KNNWarmupResponse.IndexWarmupStats(
                "index1",
                2,
                2,
                2,
                0,
                List.of(WarmupSkipReason.WARM_TIER_INDEX, WarmupSkipReason.UNKNOWN),
                Collections.emptyList()
            ),
            new KNNWarmupResponse.IndexWarmupStats("index2", 2, 2, 0, 0, Collections.emptyList(), Collections.singletonList("lucene-field"))
        );
        KNNWarmupResponse original = new KNNWarmupResponse(10, 10, 0, Collections.emptyList(), 5, indexStats);

        BytesStreamOutput out = new BytesStreamOutput();
        original.writeTo(out);
        KNNWarmupResponse deserialized = new KNNWarmupResponse(out.bytes().streamInput());
        assertEquals(original.getTotalShards(), deserialized.getTotalShards());
        assertEquals(original.getSuccessfulShards(), deserialized.getSuccessfulShards());
        assertEquals(original.getFailedShards(), deserialized.getFailedShards());
        assertEquals(original.getSkippedShards(), deserialized.getSkippedShards());
        assertEquals(original.getIndexStats(), deserialized.getIndexStats());

        // Before 3.10.0 the skip fields are not written and default to zero / empty on read
        out = new BytesStreamOutput();
        out.setVersion(Version.V_3_9_0);
        original.writeTo(out);
        try (StreamInput in = out.bytes().streamInput()) {
            in.setVersion(Version.V_3_9_0);
            deserialized = new KNNWarmupResponse(in);
        }
        assertEquals(10, deserialized.getTotalShards());
        assertEquals(10, deserialized.getSuccessfulShards());
        assertEquals(0, deserialized.getSkippedShards());
        assertTrue(deserialized.getIndexStats().isEmpty());
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

    private ClusterState clusterState() {
        return node().injector().getInstance(ClusterService.class).state();
    }

    private static Settings indexSettings(final int numberOfShards, final int numberOfReplicas) {
        return Settings.builder().put("number_of_shards", numberOfShards).put("number_of_replicas", numberOfReplicas).build();
    }

    private static KNNWarmupResponse.IndexWarmupStats getIndexStats(final KNNWarmupResponse response, final String indexName) {
        return response.getIndexStats()
            .stream()
            .filter(stats -> stats.getIndexName().equals(indexName))
            .findFirst()
            .orElseThrow(() -> new AssertionError("Expected per-index stats for [" + indexName + "]"));
    }
}
