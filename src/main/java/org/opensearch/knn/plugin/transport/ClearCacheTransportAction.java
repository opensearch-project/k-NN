/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.broadcast.node.TransportBroadcastByNodeAction;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlockException;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.cluster.routing.ShardsIterator;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.action.support.DefaultShardOperationFailedException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.index.Index;
import org.opensearch.index.IndexService;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.indices.IndicesService;
import org.opensearch.knn.index.KNNIndexShard;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

/**
 * Transport Action to evict k-NN indices from Cache. TransportBroadcastByNodeAction will distribute the request to
 * all shards across the cluster for the given indices. For each shard, shardOperation will be called and the
 * indices will be cleared from cache.
 */
public class ClearCacheTransportAction extends TransportBroadcastByNodeAction<
    ClearCacheRequest,
    ClearCacheResponse,
    TransportBroadcastByNodeAction.EmptyResult> {

    private IndicesService indicesService;

    /**
     * Constructor
     *
     * @param clusterService ClusterService
     * @param transportService TransportService
     * @param actionFilters ActionFilters
     * @param indexNameExpressionResolver IndexNameExpressionResolver
     * @param indicesService IndicesService
     */
    @Inject
    public ClearCacheTransportAction(
        ClusterService clusterService,
        TransportService transportService,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver,
        IndicesService indicesService
    ) {
        super(
            ClearCacheAction.NAME,
            clusterService,
            transportService,
            actionFilters,
            indexNameExpressionResolver,
            ClearCacheRequest::new,
            ThreadPool.Names.SEARCH
        );
        this.indicesService = indicesService;
    }

    /**
     * @param streamInput StreamInput
     * @return EmptyResult
     * @throws IOException
     */
    @Override
    protected EmptyResult readShardResult(StreamInput streamInput) throws IOException {
        return EmptyResult.readEmptyResultFrom(streamInput);
    }

    /**
     * @param request ClearCacheRequest
     * @param totalShards total number of shards on which ClearCache was performed
     * @param successfulShards number of shards that succeeded
     * @param failedShards number of shards that failed
     * @param emptyResults List of EmptyResult
     * @param shardFailures list of shard failure exceptions
     * @param clusterState ClusterState
     * @return {@link ClearCacheResponse}
     */
    @Override
    protected ClearCacheResponse newResponse(
        ClearCacheRequest request,
        int totalShards,
        int successfulShards,
        int failedShards,
        List<EmptyResult> emptyResults,
        List<DefaultShardOperationFailedException> shardFailures,
        ClusterState clusterState
    ) {
        return new ClearCacheResponse(totalShards, successfulShards, failedShards, shardFailures);
    }

    /**
     * @param streamInput StreamInput
     * @return {@link ClearCacheRequest}
     * @throws IOException
     */
    @Override
    protected ClearCacheRequest readRequestFrom(StreamInput streamInput) throws IOException {
        return new ClearCacheRequest(streamInput);
    }

    /**
     * Operation performed at a shard level on all the shards of given index where the index is removed from the cache.
     *
     * @param request ClearCacheRequest
     * @param shardRouting ShardRouting of given shard
     * @return EmptyResult
     * @throws IOException
     */
    @Override
    protected EmptyResult shardOperation(ClearCacheRequest request, ShardRouting shardRouting) throws IOException {
        Index index = shardRouting.shardId().getIndex();
        IndexService indexService = indicesService.indexServiceSafe(index);
        IndexShard indexShard = indexService.getShard(shardRouting.shardId().id());
        KNNIndexShard knnIndexShard = new KNNIndexShard(indexShard);
        knnIndexShard.clearCache();
        return EmptyResult.INSTANCE;
    }

    /**
     * @param clusterState ClusterState
     * @param request ClearCacheRequest
     * @param concreteIndices Indices in the request
     * @return ShardsIterator with all the shards for given concrete indices
     */
    @Override
    protected ShardsIterator shards(ClusterState clusterState, ClearCacheRequest request, String[] concreteIndices) {
        return clusterState.routingTable().allShards(concreteIndices);
    }

    /**
     * @param clusterState  ClusterState
     * @param request ClearCacheRequest
     * @return ClusterBlockException if there is any global cluster block at a cluster block level of "METADATA_WRITE"
     */
    @Override
    protected ClusterBlockException checkGlobalBlock(ClusterState clusterState, ClearCacheRequest request) {
        return clusterState.blocks().globalBlockedException(ClusterBlockLevel.METADATA_WRITE);
    }

    /**
     * @param clusterState ClusterState
     * @param request ClearCacheRequest
     * @param concreteIndices Indices in the request
     * @return ClusterBlockException if there is any cluster block on any of the given indices at a cluster block level of "METADATA_WRITE"
     */
    @Override
    protected ClusterBlockException checkRequestBlock(ClusterState clusterState, ClearCacheRequest request, String[] concreteIndices) {
        return clusterState.blocks().indicesBlockedException(ClusterBlockLevel.METADATA_WRITE, concreteIndices);
    }
}
