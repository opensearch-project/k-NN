/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.knn.index.KNNIndexShard;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.core.action.support.DefaultShardOperationFailedException;
import org.opensearch.action.support.broadcast.node.TransportBroadcastByNodeAction;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlockException;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.cluster.routing.ShardsIterator;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.indices.IndicesService;
import org.opensearch.transport.TransportService;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.List;

/**
 * Transport Action for warming up k-NN indices. TransportBroadcastByNodeAction will distribute the request to
 * all shards across the cluster for the given indices. For each shard, shardOperation will be called and the
 * warmup will take place.
 */
public class KNNWarmupTransportAction extends TransportBroadcastByNodeAction<
    KNNWarmupRequest,
    KNNWarmupResponse,
    TransportBroadcastByNodeAction.EmptyResult> {

    public static Logger logger = LogManager.getLogger(KNNWarmupTransportAction.class);

    private IndicesService indicesService;

    @Inject
    public KNNWarmupTransportAction(
        ClusterService clusterService,
        TransportService transportService,
        IndicesService indicesService,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        super(
            KNNWarmupAction.NAME,
            clusterService,
            transportService,
            actionFilters,
            indexNameExpressionResolver,
            KNNWarmupRequest::new,
            ThreadPool.Names.SEARCH
        );
        this.indicesService = indicesService;
    }

    @Override
    protected EmptyResult readShardResult(StreamInput in) throws IOException {
        return EmptyResult.readEmptyResultFrom(in);
    }

    @Override
    protected KNNWarmupResponse newResponse(
        KNNWarmupRequest request,
        int totalShards,
        int successfulShards,
        int failedShards,
        List<EmptyResult> emptyResults,
        List<DefaultShardOperationFailedException> shardFailures,
        ClusterState clusterState
    ) {
        return new KNNWarmupResponse(totalShards, successfulShards, failedShards, shardFailures);
    }

    @Override
    protected KNNWarmupRequest readRequestFrom(StreamInput in) throws IOException {
        return new KNNWarmupRequest(in);
    }

    @Override
    protected EmptyResult shardOperation(KNNWarmupRequest request, ShardRouting shardRouting) throws IOException {
        KNNIndexShard knnIndexShard = new KNNIndexShard(
            indicesService.indexServiceSafe(shardRouting.shardId().getIndex()).getShard(shardRouting.shardId().id())
        );
        knnIndexShard.warmup();
        return EmptyResult.INSTANCE;
    }

    @Override
    protected ShardsIterator shards(ClusterState state, KNNWarmupRequest request, String[] concreteIndices) {
        return state.routingTable().allShards(concreteIndices);
    }

    @Override
    protected ClusterBlockException checkGlobalBlock(ClusterState state, KNNWarmupRequest request) {
        return state.blocks().globalBlockedException(ClusterBlockLevel.METADATA_READ);
    }

    @Override
    protected ClusterBlockException checkRequestBlock(ClusterState state, KNNWarmupRequest request, String[] concreteIndices) {
        return state.blocks().indicesBlockedException(ClusterBlockLevel.METADATA_READ, concreteIndices);
    }
}
