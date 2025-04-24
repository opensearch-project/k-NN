/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
import org.opensearch.indices.IndicesService;
import org.opensearch.knn.index.KNNIndexShard;
import org.opensearch.transport.TransportService;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.util.List;

/**
 * Transport action for profiling KNN vectors in an index
 */
public class KNNProfileTransportAction extends TransportBroadcastByNodeAction<
    KNNProfileRequest,
    KNNProfileResponse,
    KNNProfileShardResult> {

    public static Logger logger = LogManager.getLogger(KNNProfileTransportAction.class);
    private final IndicesService indicesService;

    @Inject
    public KNNProfileTransportAction(
        ClusterService clusterService,
        TransportService transportService,
        IndicesService indicesService,
        ActionFilters actionFilters,
        IndexNameExpressionResolver indexNameExpressionResolver
    ) {
        super(
            KNNProfileAction.NAME,
            clusterService,
            transportService,
            actionFilters,
            indexNameExpressionResolver,
            KNNProfileRequest::new,
            ThreadPool.Names.SEARCH
        );
        this.indicesService = indicesService;
    }

    @Override
    protected KNNProfileShardResult readShardResult(StreamInput in) throws IOException {
        return new KNNProfileShardResult(in);
    }

    @Override
    protected KNNProfileResponse newResponse(
        KNNProfileRequest request,
        int totalShards,
        int successfulShards,
        int failedShards,
        List<KNNProfileShardResult> shardResults,
        List<DefaultShardOperationFailedException> shardFailures,
        ClusterState clusterState
    ) {
        return new KNNProfileResponse(totalShards, successfulShards, failedShards, shardResults, shardFailures);
    }

    @Override
    protected KNNProfileRequest readRequestFrom(StreamInput in) throws IOException {
        return new KNNProfileRequest(in);
    }

    @Override
    protected KNNProfileShardResult shardOperation(KNNProfileRequest request, ShardRouting shardRouting) throws IOException {
        KNNIndexShard knnIndexShard = new KNNIndexShard(
            indicesService.indexServiceSafe(shardRouting.shardId().getIndex()).getShard(shardRouting.shardId().id())
        );

        List<StatisticalSummaryValues> profileResults = knnIndexShard.profile(request.getFieldName());
        logger.info(
            "[KNN] Profile completed for field: {} on shard: {} - stats count: {}",
            request.getFieldName(),
            shardRouting.shardId(),
            profileResults != null ? profileResults.size() : 0
        );
        return new KNNProfileShardResult(shardRouting.shardId(), profileResults);
    }

    @Override
    protected ShardsIterator shards(ClusterState state, KNNProfileRequest request, String[] concreteIndices) {
        return state.routingTable().allShards(concreteIndices);
    }

    @Override
    protected ClusterBlockException checkGlobalBlock(ClusterState state, KNNProfileRequest request) {
        return state.blocks().globalBlockedException(ClusterBlockLevel.METADATA_READ);
    }

    @Override
    protected ClusterBlockException checkRequestBlock(ClusterState state, KNNProfileRequest request, String[] concreteIndices) {
        return state.blocks().indicesBlockedException(ClusterBlockLevel.METADATA_READ, concreteIndices);
    }
}
