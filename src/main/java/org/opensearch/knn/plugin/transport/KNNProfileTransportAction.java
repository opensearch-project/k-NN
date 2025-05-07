/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import lombok.extern.log4j.Log4j2;
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
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

@Log4j2
public class KNNProfileTransportAction extends TransportBroadcastByNodeAction<
    KNNProfileRequest,
    KNNProfileResponse,
    KNNIndexShardProfileResult> {

    private IndicesService indicesService;

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

    // @Override
    // protected KNNIndexShardProfileResult readShardResult(StreamInput in) throws IOException {
    // return new KNNIndexShardProfileResult(null, null);
    // }

    @Override
    protected KNNIndexShardProfileResult readShardResult(StreamInput in) throws IOException {
        return new KNNIndexShardProfileResult(in);
    }

    @Override
    protected KNNProfileResponse newResponse(
        KNNProfileRequest request,
        int totalShards,
        int successfulShards,
        int failedShards,
        List<KNNIndexShardProfileResult> profileResults,
        List<DefaultShardOperationFailedException> shardFailures,
        ClusterState clusterState
    ) {
        return new KNNProfileResponse(profileResults, totalShards, successfulShards, failedShards, shardFailures);
    }

    @Override
    protected KNNProfileRequest readRequestFrom(StreamInput in) throws IOException {
        return new KNNProfileRequest(in);
    }

    @Override
    protected KNNIndexShardProfileResult shardOperation(KNNProfileRequest request, ShardRouting shardRouting) throws IOException {
        KNNIndexShard knnIndexShard = new KNNIndexShard(
            indicesService.indexServiceSafe(shardRouting.shardId().getIndex()).getShard(shardRouting.shardId().id())
        );

        return knnIndexShard.profile(request.getField());
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
